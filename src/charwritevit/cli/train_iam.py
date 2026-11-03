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
from charwritevit.losses import glyph_classification_loss, hinge_discriminator, hinge_generator
from charwritevit.models.auxiliary import CharacterClassifier, PatchDiscriminator, WordOCR, WriterStyleNetwork
from charwritevit.train.preview import MetricHistory, save_generator_preview
from charwritevit.train.utils import amp_context, load_checkpoint, make_scaler, resolve_device, save_checkpoint, seed_everything


def _set_grad(module: torch.nn.Module, enabled: bool) -> None:
    module.requires_grad_(enabled)


def _step(loss, optimizer, scaler) -> None:
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # Do not keep parameter-gradient buffers from one network alive while a
    # different network is being trained. This matters on 16 GB Kaggle GPUs,
    # where the generator step also needs input gradients through D/OCR/Writer.
    optimizer.zero_grad(set_to_none=True)


def _ctc_loss(ctc, logits, ids, lengths):
    targets=torch.cat([ids[index,:length] for index,length in enumerate(lengths.tolist())])
    input_lengths=torch.full_like(lengths,logits.shape[1])
    return ctc(logits.float().log_softmax(-1).transpose(0,1),targets,input_lengths,lengths)


def _gradient_balance(adversarial, auxiliary, generated, coefficient=0.7):
    adversarial_gradient=torch.autograd.grad(adversarial,generated,retain_graph=True)[0]
    auxiliary_gradient=torch.autograd.grad(auxiliary,generated,retain_graph=True)[0]
    return coefficient*adversarial_gradient.std().detach()/(1e-7+auxiliary_gradient.std().detach())


def main():
    parser=training_parser("Train the character-refined WriteViT baseline on IAM")
    parser.add_argument("--classifier",required=True,help="Pretrained frozen character-classifier checkpoint")
    parser.add_argument("--ocr")
    parser.add_argument("--synthetic-checkpoint")
    # Retained only to fail loudly for old commands; WriteViT has no GAN warmup.
    parser.add_argument("--warmup-epochs",type=int)
    args=parser.parse_args()
    if args.warmup_epochs not in (None,0):
        raise ValueError("WriteViT parity mode has no GAN warmup; remove --warmup-epochs")

    cfg=load_config(args.config); seed_everything(int(cfg.get("seed",42))); device=resolve_device()
    vocab=load_vocab(cfg); train_cfg=cfg["train"]; data_cfg=cfg["data"]; loss_cfg=cfg["loss"]
    dataset=IAMWordDataset(
        Path(data_cfg["prepared_dir"])/"train.json",vocab,
        int(data_cfg.get("image_height",32)),int(data_cfg.get("char_width",16)),
        int(data_cfg.get("max_word_length",24)),int(data_cfg.get("style_references",1)),
        bool(data_cfg.get("iam_validate_images",True)),
    )
    loader=DataLoader(
        dataset,batch_size=int(train_cfg["batch_size"]),shuffle=True,
        num_workers=int(train_cfg.get("num_workers",0)),
        collate_fn=lambda batch:collate_iam(batch,vocab.pad_id),drop_last=True,
    )
    preview_manifest=Path(data_cfg["prepared_dir"])/"val.json"
    preview_dataset=IAMWordDataset(
        preview_manifest if preview_manifest.is_file() else Path(data_cfg["prepared_dir"])/"train.json",
        vocab,int(data_cfg.get("image_height",32)),int(data_cfg.get("char_width",16)),
        int(data_cfg.get("max_word_length",24)),int(data_cfg.get("style_references",1)),
        bool(data_cfg.get("iam_validate_images",True)),
    )
    if len(preview_dataset)<1:
        preview_dataset=dataset
    preview_count=min(int(train_cfg.get("preview_samples",10)),len(preview_dataset))
    if preview_count < 1: raise ValueError("No samples are available for fixed previews")
    fixed_preview_batch=next(iter(DataLoader(
        preview_dataset,batch_size=preview_count,shuffle=False,num_workers=0,
        collate_fn=lambda batch:collate_iam(batch,vocab.pad_id),
    )))
    writer_ids=sorted({record["writer_id"] for record in dataset.records})
    writer_to_id={writer:index for index,writer in enumerate(writer_ids)}

    generator=build_generator(cfg,vocab).to(device)
    if args.synthetic_checkpoint:
        generator.load_state_dict(load_checkpoint(args.synthetic_checkpoint,device)["model"])
    classifier=CharacterClassifier(len(vocab)).to(device)
    classifier_state=load_checkpoint(args.classifier,device)
    if int(classifier_state.get("classifier_data_version",0)) != 2:
        raise ValueError(
            "Character checkpoint predates uncropped EMNIST + GNU anchoring; "
            "train a fresh classifier-v2 checkpoint before IAM training"
        )
    classifier.load_state_dict(classifier_state.get("classifier",classifier_state.get("model")))
    classifier.eval(); _set_grad(classifier,False)
    ocr=WordOCR(len(vocab)).to(device)
    if args.ocr:
        ocr_state=load_checkpoint(args.ocr,device)
        ocr.load_state_dict(ocr_state.get("ocr",ocr_state.get("model")))
    discriminator=PatchDiscriminator().to(device)
    # This is the original WriteViT Writer path, shared with the generator and
    # independent from generator.character_style_encoder.
    writer=WriterStyleNetwork(
        len(writer_ids),int(cfg["model"].get("style_dim",256)),
        encoder=generator.writevit_writer_encoder,
    ).to(device)

    writer_parameter_ids={id(parameter) for parameter in generator.writevit_writer_encoder.parameters()}
    generator_parameters=[parameter for parameter in generator.parameters() if id(parameter) not in writer_parameter_ids]
    g_lr=float(train_cfg.get("generator_learning_rate",train_cfg["learning_rate"]))
    d_lr=float(train_cfg.get("discriminator_learning_rate",train_cfg["learning_rate"]))
    w_lr=float(train_cfg.get("writer_learning_rate",train_cfg["learning_rate"]))
    ocr_lr=float(train_cfg.get("ocr_learning_rate",train_cfg["learning_rate"]))
    adam_options={"betas":(0.0,0.999),"weight_decay":0.0,"eps":1e-8}
    optimizer_g=torch.optim.Adam(generator_parameters,lr=g_lr,**adam_options)
    optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=d_lr,**adam_options)
    optimizer_w=torch.optim.Adam(writer.parameters(),lr=w_lr,**adam_options)
    optimizer_ocr=torch.optim.Adam(ocr.parameters(),lr=ocr_lr,**adam_options)
    scaler_g=make_scaler(device,train_cfg.get("amp",True)); scaler_d=make_scaler(device,train_cfg.get("amp",True))
    scaler_w=make_scaler(device,train_cfg.get("amp",True)); scaler_ocr=make_scaler(device,train_cfg.get("amp",True))
    ctc=CTCLoss(blank=vocab.blank_id,zero_infinity=True)

    start_epoch=0; batch_step=0; generator_updates=0; history=MetricHistory(args.output)
    if args.resume:
        state=load_checkpoint(args.resume,device)
        if state.get("character_classifier_mode") != "pretrained_frozen":
            raise ValueError(
                "This IAM checkpoint predates the pretrained-frozen character classifier; "
                "start a fresh IAM output instead of resuming the joint-classifier run"
            )
        generator.load_state_dict(state["generator"]); discriminator.load_state_dict(state["discriminator"])
        writer.load_state_dict(state["writer"]); ocr.load_state_dict(state["ocr"])
        optimizer_g.load_state_dict(state["optimizer_g"]); optimizer_d.load_state_dict(state["optimizer_d"])
        optimizer_w.load_state_dict(state["optimizer_w"]); optimizer_ocr.load_state_dict(state["optimizer_ocr"])
        scaler_g.load_state_dict(state.get("scaler_g",{})); scaler_d.load_state_dict(state.get("scaler_d",{}))
        scaler_w.load_state_dict(state.get("scaler_w",{})); scaler_ocr.load_state_dict(state.get("scaler_ocr",{}))
        batch_step=int(state.get("batch_step",0)); generator_updates=int(state.get("generator_updates",0))
        start_epoch=int(state.get("epoch",-1))+1

    epochs=args.epochs or int(train_cfg["epochs"]); g_every=int(train_cfg.get("generator_every",2))
    log_every=max(1,int(train_cfg.get("log_every",50)))
    print(f"WriteViT parity training: samples={len(dataset)} writers={len(writer_ids)} batch={train_cfg['batch_size']} G:critic=1:{g_every}",flush=True)
    for epoch in range(start_epoch,epochs):
        totals={key:0.0 for key in ("generator","discriminator","ocr_real","writer_real","character_accuracy","character_style","ctc_fake","writer_fake","adversarial")}
        counts={key:0 for key in totals}; started=time.perf_counter()
        for batch_index,batch in enumerate(loader):
            real=batch["image"].to(device); ids=batch["ids"].to(device); references=batch["references"].to(device)
            lengths=batch["lengths"]; writer_labels=torch.tensor([writer_to_id[item] for item in batch["writer_id"]],device=device)

            # Original D/OCR/Writer critic step: every batch.
            _set_grad(discriminator,True); _set_grad(ocr,True); _set_grad(writer,True)
            with torch.no_grad():
                detached_output=generator(ids,references)
                fake_detached=detached_output["word"]
                styled_glyphs_detached=detached_output["styled_glyphs"]
            # The other no-grad outputs include all multi-scale feature maps and
            # are not used by the critic/classifier updates.
            del detached_output

            # Run the independent critic graphs one at a time. Building all four
            # graphs before the first backward unnecessarily keeps every saved
            # activation resident at once.
            with amp_context(device,train_cfg.get("amp",True)):
                real_logits=ocr(real); loss_ocr_real=_ctc_loss(ctc,real_logits,ids,lengths)
            _step(loss_ocr_real,optimizer_ocr,scaler_ocr)

            with amp_context(device,train_cfg.get("amp",True)):
                _,writer_logits=writer(real); loss_writer_real=F.cross_entropy(writer_logits,writer_labels)
            _step(loss_writer_real,optimizer_w,scaler_w)

            with amp_context(device,train_cfg.get("amp",True)):
                loss_discriminator=2.0*hinge_discriminator(discriminator(real),discriminator(fake_detached))
            _step(loss_discriminator,optimizer_d,scaler_d)

            character_mask=ids.ne(vocab.pad_id); character_labels=ids[character_mask]
            with torch.inference_mode():
                styled_logits=classifier(styled_glyphs_detached[character_mask])
                character_accuracy=(styled_logits.argmax(-1)==character_labels).float().mean()
            values={"discriminator":loss_discriminator,"ocr_real":loss_ocr_real,"writer_real":loss_writer_real,"character_accuracy":character_accuracy}
            for key,value in values.items(): totals[key]+=float(value.detach()); counts[key]+=1
            del values
            del fake_detached,styled_glyphs_detached,real_logits,writer_logits,styled_logits

            # Original generator schedule: every second batch by default.
            if batch_index % g_every == 0:
                _set_grad(discriminator,False); _set_grad(ocr,False); _set_grad(writer,False)
                with amp_context(device,train_cfg.get("amp",True)):
                    output=generator(ids,references); generated=output["word"]
                    adversarial=hinge_generator(discriminator(generated))
                    ctc_fake=_ctc_loss(ctc,ocr(generated),ids,lengths)
                    _,fake_writer_logits=writer(generated); writer_fake=F.cross_entropy(fake_writer_logits,writer_labels)
                    character_style=glyph_classification_loss(classifier,output["styled_glyphs"],ids,vocab.pad_id)
                    ocr_weight=_gradient_balance(adversarial,ctc_fake,generated)
                    writer_weight=_gradient_balance(adversarial,writer_fake,generated)
                    generator_loss=(
                        2.0*adversarial+ocr_weight*ctc_fake+writer_weight*writer_fake
                        +float(loss_cfg.get("character_style",1.0))*character_style
                    )
                _step(generator_loss,optimizer_g,scaler_g); generator_updates+=1
                values={"generator":generator_loss,"character_style":character_style,"ctc_fake":ctc_fake,"writer_fake":writer_fake,"adversarial":adversarial}
                for key,value in values.items(): totals[key]+=float(value.detach()); counts[key]+=1
                # Loop locals otherwise keep the large multi-scale output dict
                # resident throughout the following critic iteration.
                del values,output,generated,generator_loss,character_style,ctc_fake,writer_fake,adversarial,fake_writer_logits,ocr_weight,writer_weight

            batch_step+=1
            if batch_index==0 or (batch_index+1)%log_every==0:
                print(f"epoch={epoch+1}/{epochs} batch={batch_index+1}/{len(loader)} D={float(loss_discriminator):.4f} OCR={float(loss_ocr_real):.4f} W={float(loss_writer_real):.4f} C_acc={float(character_accuracy):.3f} G_updates={generator_updates}",flush=True)
            if args.max_steps and batch_step>=args.max_steps: break

        metrics={"epoch":epoch,"batch_step":batch_step,"generator_updates":generator_updates}
        metrics.update({key:totals[key]/max(1,counts[key]) for key in totals})
        history.append(metrics); history.plot("iam_curves.png")
        save_checkpoint(
            Path(args.output)/"latest.pt",generator=generator.state_dict(),discriminator=discriminator.state_dict(),
            writer=writer.state_dict(),ocr=ocr.state_dict(),classifier=classifier.state_dict(),optimizer_g=optimizer_g.state_dict(),
            optimizer_d=optimizer_d.state_dict(),optimizer_w=optimizer_w.state_dict(),optimizer_ocr=optimizer_ocr.state_dict(),
            scaler_g=scaler_g.state_dict(),scaler_d=scaler_d.state_dict(),scaler_w=scaler_w.state_dict(),scaler_ocr=scaler_ocr.state_dict(),
            writer_to_id=writer_to_id,character_classifier_mode="pretrained_frozen",classifier_data_version=2,
            epoch=epoch,batch_step=batch_step,generator_updates=generator_updates,config=cfg,
        )
        if epoch%int(train_cfg.get("preview_every",1))==0:
            generator.eval()
            with torch.inference_mode(): preview=generator(fixed_preview_batch["ids"].to(device),fixed_preview_batch["references"].to(device),noise_seed=int(cfg.get("seed",42)))
            save_generator_preview(fixed_preview_batch,preview,classifier,vocab.tokens,Path(args.output)/"preview"/f"epoch_{epoch:03d}",int(train_cfg.get("preview_samples",10)))
            generator.train()
        print(f"Epoch {epoch+1} complete in {(time.perf_counter()-started)/60:.1f} min: {metrics}",flush=True)
        if args.max_steps and batch_step>=args.max_steps: break


if __name__ == "__main__":
    main()
