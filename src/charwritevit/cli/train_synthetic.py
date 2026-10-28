from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from charwritevit.cli.common import build_generator, load_vocab, training_parser
from charwritevit.config import load_config
from charwritevit.data.synthetic import SyntheticWordDataset, collate_synthetic, discover_fonts
from charwritevit.losses import glyph_classification_loss, reconstruction_loss
from charwritevit.models.auxiliary import CharacterClassifier
from charwritevit.train.preview import MetricHistory, save_generator_preview
from charwritevit.train.utils import amp_context, load_checkpoint, make_scaler, optimizer_step, resolve_device, save_checkpoint, seed_everything


def main():
    parser=training_parser("Pretrain glyphs, context and merger on synthetic words"); parser.add_argument("--classifier",required=True); args=parser.parse_args()
    cfg=load_config(args.config); seed_everything(int(cfg.get("seed",42))); device=resolve_device(); vocab=load_vocab(cfg); tc=cfg["train"]
    ds=SyntheticWordDataset(discover_fonts(cfg["data"]["fonts_dir"],bool(cfg["data"].get("require_font_license",True))),vocab); loader=DataLoader(ds,batch_size=int(tc["batch_size"]),shuffle=True,num_workers=int(tc.get("num_workers",2)),collate_fn=lambda b:collate_synthetic(b,vocab.pad_id))
    model=build_generator(cfg,vocab).to(device); classifier=CharacterClassifier(len(vocab)).to(device)
    classifier.load_state_dict(load_checkpoint(args.classifier,device)["model"]); classifier.eval(); classifier.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=float(tc["learning_rate"])); scaler=make_scaler(device,tc.get("amp",True)); step=0; start_epoch=0; history=MetricHistory(args.output); lc=cfg.get("loss",{})
    if args.resume:
        state=load_checkpoint(args.resume,device); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); scaler.load_state_dict(state.get("scaler",{})); step=int(state.get("step",0)); start_epoch=int(state.get("epoch",-1))+1
    for epoch in range(start_epoch, args.epochs or int(tc["epochs"])):
        totals={"loss":0.0,"reconstruction":0.0,"edge":0.0,"glyph_pixel":0.0,"raw_character":0.0,"context_character":0.0}; seen=0; preview_batch=None
        for batch in loader:
            ids,refs,target,glyph_target=[batch[x].to(device) for x in ("ids","references","word","glyphs")]
            with amp_context(device,tc.get("amp",True)):
                out=model(ids,refs); pixel,edge=reconstruction_loss(out["word"],target)
                raw_pixel=F.l1_loss(out["raw_glyphs"],glyph_target); char=glyph_classification_loss(classifier,out["raw_glyphs"],ids,vocab.pad_id)
                char_ctx=glyph_classification_loss(classifier,out["contextual_glyphs"],ids,vocab.pad_id)
                loss=float(lc.get("reconstruction",10))*pixel+float(lc.get("edge",0))*edge+5*raw_pixel+float(lc.get("character_raw",1))*char+float(lc.get("character_context",.25))*char_ctx
            size=ids.shape[0]; seen+=size
            for key,value in {"loss":loss,"reconstruction":pixel,"edge":edge,"glyph_pixel":raw_pixel,"raw_character":char,"context_character":char_ctx}.items(): totals[key]+=float(value.detach())*size
            preview_batch=batch
            optimizer_step(loss,opt,scaler,int(tc.get("accumulation_steps",1)),step); step+=1
            if args.max_steps and step>=args.max_steps: break
        save_checkpoint(Path(args.output)/"latest.pt",model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,step=step,config=cfg)
        metrics={"epoch":epoch,"step":step,**{key:value/max(1,seen) for key,value in totals.items()},"learning_rate":opt.param_groups[0]["lr"]}; history.append(metrics); history.plot("synthetic_curves.png")
        if preview_batch is not None and epoch%int(tc.get("preview_every",1))==0:
            model.eval()
            with torch.inference_mode(): preview_out=model(preview_batch["ids"].to(device),preview_batch["references"].to(device),noise_seed=int(cfg.get("seed",42)))
            save_generator_preview(preview_batch,preview_out,classifier,vocab.tokens,Path(args.output)/"preview"/f"epoch_{epoch:03d}",min(4,int(tc.get("preview_samples",16)))); model.train()
        print(metrics)
        if args.max_steps and step>=args.max_steps: break


if __name__ == "__main__": main()
