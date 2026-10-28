from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


class MetricHistory:
    def __init__(self, output: str | Path):
        self.root=Path(output); self.root.mkdir(parents=True,exist_ok=True); self.path=self.root/"metrics.jsonl"; self.rows=[]
        if self.path.is_file():
            self.rows=[json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, row: dict) -> None:
        clean={key:(float(value) if isinstance(value,(float,int)) else value) for key,value in row.items()}
        self.rows.append(clean)
        with self.path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(clean)+"\n")

    def plot(self, filename="losses.png") -> None:
        if not self.rows: return
        numeric=defaultdict(list); epochs=[]
        for row in self.rows:
            epochs.append(row.get("epoch",len(epochs)))
            for key,value in row.items():
                if key not in {"epoch","step"} and isinstance(value,(float,int)): numeric[key].append(value)
        figure,axis=plt.subplots(figsize=(10,6))
        for key,values in numeric.items():
            if len(values)==len(epochs): axis.plot(epochs,values,label=key)
        axis.set_xlabel("epoch"); axis.grid(alpha=.25); axis.legend(); figure.tight_layout(); figure.savefig(self.root/filename,dpi=150); plt.close(figure)


def save_classifier_preview(images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor, tokens: list[str], path: str | Path, limit=16) -> None:
    count=min(limit,images.shape[0]); columns=4; rows=(count+columns-1)//columns
    figure,_=plt.subplots(rows,columns,figsize=(columns*3,rows*3)); axes=figure.axes
    probabilities=logits.softmax(-1); confidence,prediction=probabilities.max(-1)
    for index,axis in enumerate(axes):
        axis.axis("off")
        if index>=count: continue
        axis.imshow(((images[index,0].detach().cpu()+1)*127.5).clamp(0,255),cmap="gray",vmin=0,vmax=255)
        truth=tokens[int(labels[index])]; pred=tokens[int(prediction[index])]
        axis.set_title(f"true={truth} pred={pred}\np={float(confidence[index]):.3f}")
    figure.tight_layout(); target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); figure.savefig(target,dpi=150); plt.close(figure)


def _show(axis, tensor: torch.Tensor, title: str) -> None:
    axis.imshow(((tensor.detach().cpu().squeeze()+1)*127.5).clamp(0,255),cmap="gray",vmin=0,vmax=255,aspect="auto")
    axis.set_title(title); axis.axis("off")


def save_generator_preview(batch: dict, output: dict, classifier, tokens: list[str], root: str | Path, limit=4) -> None:
    root=Path(root); root.mkdir(parents=True,exist_ok=True); count=min(limit,output["raw_glyphs"].shape[0])
    for sample in range(count):
        length=int(batch["lengths"][sample]); ids=batch["ids"][sample,:length]
        raw=output["raw_glyphs"][sample,:length]; context=output["contextual_glyphs"][sample,:length]
        with torch.inference_mode(): raw_logits=classifier(raw); context_logits=classifier(context)
        figure,axes=plt.subplots(2,max(1,length),figsize=(max(1,length)*2.2,4.5),squeeze=False)
        for index in range(length):
            truth=tokens[int(ids[index])]; raw_pred=tokens[int(raw_logits[index].argmax())]; ctx_pred=tokens[int(context_logits[index].argmax())]
            _show(axes[0,index],raw[index],f"{truth} raw→{raw_pred}"); _show(axes[1,index],context[index],f"{truth} ctx→{ctx_pred}")
        figure.tight_layout(); figure.savefig(root/f"sample_{sample:02d}_characters.png",dpi=150); plt.close(figure)
        target=batch.get("image",batch.get("word"))[sample]
        references=batch["references"][sample]
        figure,axes=plt.subplots(4,1,figsize=(12,7))
        _show(axes[0],references[0],"reference"); _show(axes[1],target,"target"); _show(axes[2],output["coarse_word"][sample],"multi-scale output"); _show(axes[3],output["word"][sample],"final word")
        figure.tight_layout(); figure.savefig(root/f"sample_{sample:02d}_word.png",dpi=150); plt.close(figure)
