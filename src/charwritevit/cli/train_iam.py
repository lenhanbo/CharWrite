from __future__ import annotations

from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from charwritevit.cli.common import build_generator, load_vocab, training_parser
from charwritevit.config import load_config
from charwritevit.data.iam import IAMWordDataset, collate_iam
from charwritevit.losses import glyph_classification_loss, hinge_discriminator, hinge_generator, reconstruction_loss
from charwritevit.models.auxiliary import CharacterClassifier, PatchDiscriminator, WordOCR, WriterStyleNetwork
from charwritevit.train.preview import MetricHistory, save_generator_preview
from charwritevit.train.utils import amp_context, load_checkpoint, make_scaler, optimizer_step, resolve_device, save_checkpoint, seed_everything


def _freeze(module): module.eval(); module.requires_grad_(False); return module
def _grad(module, enabled): module.requires_grad_(enabled)


def main():
    parser=training_parser("Train character-augmented WriteViT end-to-end on IAM"); parser.add_argument("--classifier",required=True); parser.add_argument("--ocr"); parser.add_argument("--synthetic-checkpoint"); parser.add_argument("--warmup-epochs",type=int,default=5)
    args=parser.parse_args(); cfg=load_config(args.config); seed_everything(int(cfg.get("seed",42))); device=resolve_device(); vocab=load_vocab(cfg); tc=cfg["train"]; dc=cfg["data"]; lc=cfg["loss"]
    print(f"Initializing IAM training on {device} with config={args.config}",flush=True)
    ds=IAMWordDataset(Path(dc["prepared_dir"])/"train.json",vocab,int(dc.get("image_height",32)),int(dc.get("char_width",16)),int(dc.get("max_word_length",24)),int(dc.get("style_references",4)),bool(dc.get("iam_validate_images",True)))
    loader=DataLoader(ds,batch_size=int(tc["batch_size"]),shuffle=True,num_workers=int(tc.get("num_workers",2)),collate_fn=lambda b:collate_iam(b,vocab.pad_id),drop_last=True)
    writer_ids=sorted({r["writer_id"] for r in ds.records}); writer_to_id={x:i for i,x in enumerate(writer_ids)}
    print(f"IAM dataset ready: samples={len(ds)}, writers={len(writer_ids)}, batches/epoch={len(loader)}",flush=True)
    print("Building generator and auxiliary networks",flush=True)
    gen=build_generator(cfg,vocab).to(device)
    if args.synthetic_checkpoint: gen.load_state_dict(load_checkpoint(args.synthetic_checkpoint,device)["model"])
    print(f"Loading character classifier: {args.classifier}",flush=True)
    classifier=_freeze(CharacterClassifier(len(vocab)).to(device)); classifier.load_state_dict(load_checkpoint(args.classifier,device)["model"])
    ocr=WordOCR(len(vocab)).to(device)
    if args.ocr:
        ocr_state=load_checkpoint(args.ocr,device); ocr.load_state_dict(ocr_state.get("ocr",ocr_state.get("model")))
    disc=PatchDiscriminator().to(device); style=WriterStyleNetwork(len(writer_ids),int(cfg["model"].get("style_dim",128))).to(device)
    print("Networks ready",flush=True)
    opt_g=torch.optim.AdamW(gen.parameters(),lr=float(tc["learning_rate"])); opt_d=torch.optim.AdamW(list(disc.parameters())+list(style.parameters()),lr=float(tc["learning_rate"])); opt_ocr=torch.optim.AdamW(ocr.parameters(),lr=float(tc.get("ocr_learning_rate",tc["learning_rate"]))); scaler_g=make_scaler(device,tc.get("amp",True)); scaler_d=make_scaler(device,tc.get("amp",True)); scaler_ocr=make_scaler(device,tc.get("amp",True)); ctc=CTCLoss(blank=vocab.blank_id,zero_infinity=True)
    step=0; accumulation=int(tc.get("accumulation_steps",1)); start_epoch=0; history=MetricHistory(args.output); log_every=max(1,int(tc.get("log_every",50)))
    if args.resume:
        state=load_checkpoint(args.resume,device); gen.load_state_dict(state["generator"]); disc.load_state_dict(state["discriminator"]); style.load_state_dict(state["style"]); ocr.load_state_dict(state["ocr"]); opt_g.load_state_dict(state["optimizer_g"]); opt_d.load_state_dict(state["optimizer_d"]); opt_ocr.load_state_dict(state["optimizer_ocr"]); scaler_g.load_state_dict(state.get("scaler_g",{})); scaler_d.load_state_dict(state.get("scaler_d",{})); scaler_ocr.load_state_dict(state.get("scaler_ocr",{})); step=int(state.get("step",0)); start_epoch=int(state.get("epoch",-1))+1
    total_epochs=args.epochs or int(tc["epochs"])
    print(
        f"Starting IAM training: device={device}, samples={len(ds)}, "
        f"batches/epoch={len(loader)}, batch_size={tc['batch_size']}, "
        f"accumulation={accumulation}, epochs={start_epoch}..{total_epochs-1}, "
        f"log_every={log_every}, output={args.output}",
        flush=True,
    )
    for epoch in range(start_epoch, total_epochs):
        adversarial=epoch>=args.warmup_epochs; totals={"generator":0.0,"discriminator":0.0,"ocr_real":0.0,"reconstruction":0.0,"edge":0.0,"raw_character":0.0,"context_character":0.0,"ctc":0.0,"style":0.0,"adversarial":0.0}; seen=0; preview_batch=None
        epoch_started=time.perf_counter(); epoch_batches=len(loader)
        print(f"Epoch {epoch+1}/{total_epochs} started (GAN={'on' if adversarial else 'off'})",flush=True)
        for batch_index,batch in enumerate(loader,1):
            real,ids,refs=batch["image"].to(device),batch["ids"].to(device),batch["references"].to(device); lengths=batch["lengths"]; writer=torch.tensor([writer_to_id[x] for x in batch["writer_id"]],device=device)
            targets=torch.cat([ids[i,:n] for i,n in enumerate(lengths.tolist())])
            _grad(ocr,True); ocr.train()
            with amp_context(device,tc.get("amp",True)):
                real_logits=ocr(real); real_in_len=(lengths*2).clamp(max=real_logits.shape[1]); loss_ocr_real=ctc(real_logits.log_softmax(-1).transpose(0,1),targets,real_in_len,lengths)
            optimizer_step(loss_ocr_real,opt_ocr,scaler_ocr,accumulation,step)
            with torch.no_grad(): fake_detached=gen(ids,refs)["word"]
            _grad(disc,True); _grad(style,True)
            with amp_context(device,tc.get("amp",True)):
                _,style_logits=style(real); loss_d=F.cross_entropy(style_logits,writer)
                if adversarial: loss_d=loss_d+hinge_discriminator(disc(real),disc(fake_detached))
            optimizer_step(loss_d,opt_d,scaler_d,accumulation,step)
            _grad(disc,False); _grad(style,False); _grad(ocr,False); ocr.eval()
            with amp_context(device,tc.get("amp",True)):
                out=gen(ids,refs); pixel,edge=reconstruction_loss(out["word"],real)
                raw_char=glyph_classification_loss(classifier,out["raw_glyphs"],ids,vocab.pad_id); ctx_char=glyph_classification_loss(classifier,out["contextual_glyphs"],ids,vocab.pad_id)
                logits=ocr(out["word"]); in_len=(lengths*2).clamp(max=logits.shape[1]); loss_ctc=ctc(logits.log_softmax(-1).transpose(0,1),targets,in_len,lengths)
                fake_style,fake_writer=style(out["word"]); ref_style,_=style(refs[:,0]); style_loss=F.cross_entropy(fake_writer,writer)+F.cosine_embedding_loss(fake_style,ref_style,torch.ones(real.shape[0],device=device))
                loss_g=float(lc["reconstruction"])*pixel+float(lc["edge"])*edge+float(lc["character_raw"])*raw_char+float(lc["character_context"])*ctx_char+float(lc["ctc"])*loss_ctc+float(lc["style"])*style_loss
                loss_adv=hinge_generator(disc(out["word"])) if adversarial else out["word"].new_zeros(())
                loss_g=loss_g+float(lc["adversarial"])*loss_adv
            size=ids.shape[0]; seen+=size
            values={"generator":loss_g,"discriminator":loss_d,"ocr_real":loss_ocr_real,"reconstruction":pixel,"edge":edge,"raw_character":raw_char,"context_character":ctx_char,"ctc":loss_ctc,"style":style_loss,"adversarial":loss_adv}
            for key,value in values.items(): totals[key]+=float(value.detach())*size
            preview_batch=batch
            optimizer_step(loss_g,opt_g,scaler_g,accumulation,step); step+=1
            if batch_index==1 or batch_index%log_every==0 or batch_index==epoch_batches or (args.max_steps and step>=args.max_steps):
                elapsed=time.perf_counter()-epoch_started; seconds_per_batch=elapsed/batch_index; remaining=max(0,epoch_batches-batch_index)
                if args.max_steps: remaining=min(remaining,max(0,args.max_steps-step))
                print(
                    f"epoch={epoch+1}/{total_epochs} batch={batch_index}/{epoch_batches} step={step} "
                    f"g={float(loss_g.detach()):.4f} d={float(loss_d.detach()):.4f} "
                    f"ocr_real={float(loss_ocr_real.detach()):.4f} recon={float(pixel.detach()):.4f} "
                    f"char_raw={float(raw_char.detach()):.4f} char_ctx={float(ctx_char.detach()):.4f} "
                    f"ctc={float(loss_ctc.detach()):.4f} style={float(style_loss.detach()):.4f} "
                    f"sec/batch={seconds_per_batch:.2f} eta_min={remaining*seconds_per_batch/60:.1f}",
                    flush=True,
                )
            if args.max_steps and step>=args.max_steps: break
        print(f"Saving checkpoint to {Path(args.output)/'latest.pt'}",flush=True)
        save_checkpoint(Path(args.output)/"latest.pt",generator=gen.state_dict(),discriminator=disc.state_dict(),style=style.state_dict(),ocr=ocr.state_dict(),optimizer_g=opt_g.state_dict(),optimizer_d=opt_d.state_dict(),optimizer_ocr=opt_ocr.state_dict(),scaler_g=scaler_g.state_dict(),scaler_d=scaler_d.state_dict(),scaler_ocr=scaler_ocr.state_dict(),writer_to_id=writer_to_id,epoch=epoch,step=step,config=cfg)
        metrics={"epoch":epoch,"step":step,**{key:value/max(1,seen) for key,value in totals.items()},"learning_rate":opt_g.param_groups[0]["lr"],"gan_enabled":int(adversarial)}; history.append(metrics); history.plot("iam_curves.png")
        if preview_batch is not None and epoch%int(tc.get("preview_every",1))==0:
            print(f"Writing preview for epoch {epoch+1}",flush=True)
            gen.eval()
            with torch.inference_mode(): preview_out=gen(preview_batch["ids"].to(device),preview_batch["references"].to(device),noise_seed=int(cfg.get("seed",42)))
            save_generator_preview(preview_batch,preview_out,classifier,vocab.tokens,Path(args.output)/"preview"/f"epoch_{epoch:03d}",min(4,int(tc.get("preview_samples",16)))); gen.train()
        print(f"Epoch {epoch+1} complete in {(time.perf_counter()-epoch_started)/60:.1f} min: {metrics}",flush=True)
        if args.max_steps and step>=args.max_steps: break


if __name__ == "__main__": main()

