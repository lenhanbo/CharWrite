from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

from charwritevit.cli.common import load_vocab, training_parser
from charwritevit.config import load_config
from charwritevit.data.iam import IAMWordDataset, collate_iam
from charwritevit.models.auxiliary import WordOCR
from charwritevit.train.utils import amp_context, make_scaler, optimizer_step, resolve_device, save_checkpoint, seed_everything


def main():
    args=training_parser("Train IAM word OCR").parse_args(); cfg=load_config(args.config); seed_everything(int(cfg.get("seed",42)))
    device=resolve_device(); vocab=load_vocab(cfg); tc=cfg["train"]; dc=cfg["data"]
    ds=IAMWordDataset(Path(dc["prepared_dir"])/"train.json",vocab,int(dc.get("image_height",32)),int(dc.get("char_width",16)),int(dc.get("max_word_length",24)),int(dc.get("style_references",4)))
    loader=DataLoader(ds,batch_size=int(tc["batch_size"]),shuffle=True,num_workers=int(tc.get("num_workers",2)),collate_fn=lambda b:collate_iam(b,vocab.pad_id))
    model=WordOCR(len(vocab)).to(device); opt=torch.optim.AdamW(model.parameters(),lr=float(tc["learning_rate"])); scaler=make_scaler(device,tc.get("amp",True)); criterion=CTCLoss(blank=vocab.blank_id,zero_infinity=True)
    step=0; start_epoch=0
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); scaler.load_state_dict(state.get("scaler",{})); step=int(state.get("step",0)); start_epoch=int(state.get("epoch",-1))+1
    for epoch in range(start_epoch, args.epochs or int(tc["epochs"])):
        for batch in loader:
            image,ids,lengths=batch["image"].to(device),batch["ids"].to(device),batch["lengths"]
            targets=torch.cat([ids[i,:n] for i,n in enumerate(lengths.tolist())])
            with amp_context(device,tc.get("amp",True)):
                logits=model(image); input_lengths=(lengths*2).clamp(max=logits.shape[1]); loss=criterion(logits.log_softmax(-1).transpose(0,1),targets,input_lengths,lengths)
            optimizer_step(loss,opt,scaler,int(tc.get("accumulation_steps",1)),step); step+=1
            if args.max_steps and step>=args.max_steps: break
        save_checkpoint(Path(args.output)/"latest.pt",model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,step=step,config=cfg)
        print({"epoch":epoch,"step":step,"loss":float(loss.detach())})
        if args.max_steps and step>=args.max_steps: break


if __name__ == "__main__": main()

