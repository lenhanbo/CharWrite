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
from charwritevit.train.utils import load_checkpoint, resolve_device, seed_everything


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


def _inception_images(images: torch.Tensor, fixed_width: int) -> torch.Tensor:
    """Pad variable-width grayscale words consistently and convert to RGB [0, 1]."""
    width=min(images.shape[-1],fixed_width)
    canvas=torch.ones(images.shape[0],1,images.shape[-2],fixed_width,device=images.device,dtype=images.dtype)
    canvas[...,:width]=images[...,:width]
    return ((canvas+1.0)*0.5).clamp(0,1).repeat(1,3,1,1)


def _make_distribution_metrics(device, kid_subset_size: int, weights_path: str | None = None):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.kid import KernelInceptionDistance
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("FID/KID require torchmetrics and torch-fidelity; reinstall with `pip install -e .`") from error
    weight_options={"feature_extractor_weights_path":weights_path} if weights_path else {}
    fid=FrechetInceptionDistance(feature=2048,normalize=True,**weight_options).to(device)
    kid=KernelInceptionDistance(feature=2048,subset_size=kid_subset_size,normalize=True,**weight_options).to(device)
    return fid,kid


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint",required=True); p.add_argument("--classifier"); p.add_argument("--ocr"); p.add_argument("--output",required=True); p.add_argument("--max-batches",type=int,default=100); p.add_argument("--kid-subset-size",type=int,default=100); p.add_argument("--inception-weights"); p.add_argument("--skip-fid-kid",action="store_true"); args=p.parse_args()
    cfg=load_config(args.config); seed_everything(int(cfg.get("seed",42))); device=resolve_device(); vocab=load_vocab(cfg); dc=cfg["data"]
    ds=IAMWordDataset(Path(dc["prepared_dir"])/"test.json",vocab,references=int(dc.get("style_references",1)),validate_paths=bool(dc.get("iam_validate_images",True))); loader=DataLoader(ds,batch_size=8,collate_fn=lambda b:collate_iam(b,vocab.pad_id))
    maximum_samples=min(len(ds),args.max_batches*8)
    if not args.skip_fid_kid and maximum_samples<2: raise ValueError("FID/KID require at least two evaluation samples")
    kid_subset_size=min(args.kid_subset_size,maximum_samples)
    fid=kid=None
    if not args.skip_fid_kid:
        print(f"Initializing Inception FID/KID (samples<={maximum_samples}, KID subset={kid_subset_size})",flush=True)
        fid,kid=_make_distribution_metrics(device,kid_subset_size,args.inception_weights)
    model=build_generator(cfg,vocab).to(device); state=load_checkpoint(args.checkpoint,device); model.load_state_dict(state.get("generator",state.get("model"))); model.eval()
    classifier=CharacterClassifier(len(vocab)).to(device)
    if args.classifier:
        classifier_state=load_checkpoint(args.classifier,device); classifier.load_state_dict(classifier_state.get("classifier",classifier_state.get("model")))
    elif "classifier" in state: classifier.load_state_dict(state["classifier"])
    else: raise ValueError("Checkpoint has no joint character classifier; provide --classifier for a legacy run")
    classifier.eval()
    ocr=WordOCR(len(vocab)).to(device)
    if args.ocr:
        ocr_state=load_checkpoint(args.ocr,device); ocr.load_state_dict(ocr_state.get("ocr",ocr_state.get("model")))
    elif "ocr" in state: ocr.load_state_dict(state["ocr"])
    else: raise ValueError("Checkpoint has no joint OCR state; provide --ocr for a legacy checkpoint")
    ocr.eval()
    writer_map=state.get("writer_to_id",{}); style=WriterStyleNetwork(max(1,len(writer_map)),int(cfg["model"].get("style_dim",256)),encoder=model.writevit_writer_encoder).to(device)
    if "writer" in state: style.load_state_dict(state["writer"])
    elif "style" in state: style.load_state_dict(state["style"])
    style.eval(); pixels=[]; edges=[]; ssims=[]; raw_correct=ctx_correct=char_total=edits=characters=words=exact=style_hits=style_total=evaluated_samples=0
    fixed_width=int(dc.get("char_width",16))*int(dc.get("max_word_length",24))
    with torch.inference_mode():
        for index,batch in enumerate(loader):
            if index>=args.max_batches: break
            ids=batch["ids"].to(device); refs=batch["references"].to(device); target=batch["image"].to(device); out=model(ids,refs); pixel,edge=reconstruction_loss(out["word"],target); pixels.append(float(pixel)); edges.append(float(edge)); ssims.append(float(_ssim((out["word"]+1)/2,(target+1)/2)))
            if fid is not None and kid is not None:
                real_inception=_inception_images(target,fixed_width); fake_inception=_inception_images(out["word"],fixed_width)
                fid.update(real_inception,real=True); fid.update(fake_inception,real=False)
                kid.update(real_inception,real=True); kid.update(fake_inception,real=False)
            evaluated_samples+=ids.shape[0]
            mask=ids.ne(vocab.pad_id); raw_correct+=(classifier(out["gnu_glyphs"][mask]).argmax(-1)==ids[mask]).sum().item(); ctx_correct+=(classifier(out["styled_glyphs"][mask]).argmax(-1)==ids[mask]).sum().item(); char_total+=mask.sum().item()
            predictions=_decode_ctc(ocr(out["word"]),vocab)
            for prediction,truth in zip(predictions,batch["text"]): edits+=_edit_distance(prediction,truth); characters+=len(truth); words+=1; exact+=prediction==truth
            generated=F.normalize(style.encoder(out["word"]).mean(1),dim=-1); reference=F.normalize(style.encoder(refs).mean(1),dim=-1); similarity=generated@reference.T; style_hits+=(similarity.argmax(-1)==torch.arange(similarity.shape[0],device=device)).sum().item(); style_total+=similarity.shape[0]
    metrics={"foreground_l1":sum(pixels)/max(1,len(pixels)),"edge_l1":sum(edges)/max(1,len(edges)),"ssim":sum(ssims)/max(1,len(ssims)),"gnu_character_accuracy":raw_correct/max(1,char_total),"styled_character_accuracy":ctx_correct/max(1,char_total),"cer":edits/max(1,characters),"word_accuracy":exact/max(1,words),"style_retrieval_at_1":style_hits/max(1,style_total),"batches":len(pixels),"samples":evaluated_samples}
    if fid is not None and kid is not None:
        kid_mean,kid_std=kid.compute(); metrics.update({"fid":float(fid.compute()),"kid_mean":float(kid_mean),"kid_std":float(kid_std),"kid_subset_size":kid_subset_size})
    target=Path(args.output); target.mkdir(parents=True,exist_ok=True); (target/"metrics.json").write_text(json.dumps(metrics,indent=2)); print(metrics)


if __name__ == "__main__": main()
