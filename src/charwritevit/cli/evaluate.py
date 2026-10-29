from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from charwritevit.cli.common import build_generator, load_vocab
from charwritevit.config import load_config
from charwritevit.data.iam import IAMWordDataset, collate_iam
from charwritevit.losses import reconstruction_loss
from charwritevit.models.auxiliary import CharacterClassifier, WordOCR, WriterStyleNetwork
from charwritevit.train.utils import load_checkpoint, resolve_device


def _edit_distance(a: str, b: str) -> int:
    row=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        new=[i]
        for j,cb in enumerate(b,1): new.append(min(new[-1]+1,row[j]+1,row[j-1]+(ca!=cb)))
        row=new
    return row[-1]


def _decode_ctc(logits, vocab):
    results=[]
    for sequence in logits.argmax(-1).tolist():
        collapsed=[]; previous=None
        for token in sequence:
            if token!=previous and token!=vocab.blank_id: collapsed.append(token)
            previous=token
        results.append(vocab.decode(collapsed))
    return results


def _ssim(x, y):
    c1,c2=.01**2,.03**2; mx=x.mean((-2,-1)); my=y.mean((-2,-1)); vx=x.var((-2,-1),unbiased=False); vy=y.var((-2,-1),unbiased=False); cov=((x-mx[...,None,None])*(y-my[...,None,None])).mean((-2,-1))
    return (((2*mx*my+c1)*(2*cov+c2))/((mx.square()+my.square()+c1)*(vx+vy+c2))).mean()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--classifier",required=True); p.add_argument("--ocr"); p.add_argument("--output",required=True); p.add_argument("--max-batches",type=int,default=100); args=p.parse_args()
    cfg=load_config(args.config); device=resolve_device(); vocab=load_vocab(cfg); dc=cfg["data"]
    ds=IAMWordDataset(Path(dc["prepared_dir"])/"test.json",vocab,references=int(dc.get("style_references",4))); loader=DataLoader(ds,batch_size=8,collate_fn=lambda b:collate_iam(b,vocab.pad_id))
    model=build_generator(cfg,vocab).to(device); state=load_checkpoint(args.checkpoint,device); model.load_state_dict(state.get("generator",state.get("model"))); model.eval()
    classifier=CharacterClassifier(len(vocab)).to(device); classifier.load_state_dict(load_checkpoint(args.classifier,device)["model"]); classifier.eval()
    ocr=WordOCR(len(vocab)).to(device)
    if args.ocr:
        ocr_state=load_checkpoint(args.ocr,device); ocr.load_state_dict(ocr_state.get("ocr",ocr_state.get("model")))
    elif "ocr" in state: ocr.load_state_dict(state["ocr"])
    else: raise ValueError("Checkpoint has no joint OCR state; provide --ocr for a legacy checkpoint")
    ocr.eval()
    writer_map=state.get("writer_to_id",{}); style=WriterStyleNetwork(max(1,len(writer_map)),int(cfg["model"].get("style_dim",128))).to(device)
    if "style" in state: style.load_state_dict(state["style"])
    style.eval(); pixels=[]; edges=[]; ssims=[]; raw_correct=ctx_correct=char_total=edits=characters=words=exact=style_hits=style_total=0
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            if index>=args.max_batches: break
            ids=batch["ids"].to(device); refs=batch["references"].to(device); target=batch["image"].to(device); out=model(ids,refs); pixel,edge=reconstruction_loss(out["word"],target); pixels.append(float(pixel)); edges.append(float(edge)); ssims.append(float(_ssim((out["word"]+1)/2,(target+1)/2)))
            mask=ids.ne(vocab.pad_id); raw_correct+=(classifier(out["raw_glyphs"][mask]).argmax(-1)==ids[mask]).sum().item(); ctx_correct+=(classifier(out["contextual_glyphs"][mask]).argmax(-1)==ids[mask]).sum().item(); char_total+=mask.sum().item()
            predictions=_decode_ctc(ocr(out["word"]),vocab)
            for prediction,truth in zip(predictions,batch["text"]): edits+=_edit_distance(prediction,truth); characters+=len(truth); words+=1; exact+=prediction==truth
            generated=F.normalize(style.encoder(out["word"]).mean(1),dim=-1); reference=F.normalize(style.encoder(refs).mean(1),dim=-1); similarity=generated@reference.T; style_hits+=(similarity.argmax(-1)==torch.arange(similarity.shape[0],device=device)).sum().item(); style_total+=similarity.shape[0]
    metrics={"foreground_l1":sum(pixels)/max(1,len(pixels)),"edge_l1":sum(edges)/max(1,len(edges)),"ssim":sum(ssims)/max(1,len(ssims)),"raw_character_accuracy":raw_correct/max(1,char_total),"context_character_accuracy":ctx_correct/max(1,char_total),"cer":edits/max(1,characters),"word_accuracy":exact/max(1,words),"style_retrieval_at_1":style_hits/max(1,style_total),"batches":len(pixels)}
    target=Path(args.output); target.mkdir(parents=True,exist_ok=True); (target/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(metrics)


if __name__ == "__main__": main()
