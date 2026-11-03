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
                if key not in {"epoch","step","batch_step","generator_updates"} and isinstance(value,(float,int)): numeric[key].append(value)
        figure,axis=plt.subplots(figsize=(10,6))
        for key,values in numeric.items():
            if len(values)==len(epochs): axis.plot(epochs,values,label=key)
        axis.set_xlabel("epoch"); axis.grid(alpha=.25); axis.legend(); figure.tight_layout(); figure.savefig(self.root/filename,dpi=150); plt.close(figure)


def save_classifier_preview(images: torch.Tensor, labels: torch.Tensor, logits: torch.Tensor,
                            tokens: list[str], path: str | Path, limit=16,
                            columns=4, title: str | None = None) -> None:
    count=min(limit,images.shape[0]); columns=min(columns,max(1,count)); rows=(count+columns-1)//columns
    figure,_=plt.subplots(rows,columns,figsize=(columns*2.2,rows*2.2),squeeze=False); axes=figure.axes
    probabilities=logits.softmax(-1); confidence,prediction=probabilities.max(-1)
    for index,axis in enumerate(axes):
        axis.axis("off")
        if index>=count: continue
        axis.imshow(((images[index,0].detach().cpu()+1)*127.5).clamp(0,255),cmap="gray",vmin=0,vmax=255)
        truth=tokens[int(labels[index])]; pred=tokens[int(prediction[index])]
        axis.set_title(f"true={truth} pred={pred}\np={float(confidence[index]):.3f}",fontsize=8)
    if title: figure.suptitle(title)
    figure.tight_layout(); target=Path(path); target.parent.mkdir(parents=True,exist_ok=True); figure.savefig(target,dpi=150); plt.close(figure)


def _show(axis, tensor: torch.Tensor, title: str) -> None:
    axis.imshow(((tensor.detach().cpu().squeeze()+1)*127.5).clamp(0,255),cmap="gray",vmin=0,vmax=255,aspect="auto")
    axis.set_title(title); axis.axis("off")


def _show_feature(axis, tensor: torch.Tensor, title: str, width: int) -> None:
    """Render a multi-channel feature honestly as normalized channel energy."""
    feature=tensor.detach().float().cpu()[..., :width]
    energy=feature.abs().mean(0)
    minimum,maximum=energy.min(),energy.max()
    energy=(energy-minimum)/(maximum-minimum+1e-8)
    axis.imshow(energy,cmap="magma",vmin=0,vmax=1,aspect="auto")
    axis.set_title(f"{title} | mean(abs(channel)), C/H/W={tuple(tensor.shape)}")
    axis.axis("off")


def save_generator_preview(batch: dict, output: dict, classifier, tokens: list[str], root: str | Path, limit=10) -> None:
    root=Path(root); root.mkdir(parents=True,exist_ok=True); count=min(limit,output["gnu_glyphs"].shape[0])
    full_length=output["gnu_glyphs"].shape[1]
    for sample in range(count):
        length=int(batch["lengths"][sample]); ids=batch["ids"][sample,:length]
        gnu=output["gnu_glyphs"][sample,:length]; styled=output["styled_glyphs"][sample,:length]
        with torch.inference_mode(): gnu_logits=classifier(gnu); styled_logits=classifier(styled)

        figure,axes=plt.subplots(2,max(1,length),figsize=(max(1,length)*2.2,4.5),squeeze=False)
        for index in range(length):
            truth=tokens[int(ids[index])]; gnu_pred=tokens[int(gnu_logits[index].argmax())]; styled_pred=tokens[int(styled_logits[index].argmax())]
            _show(axes[0,index],gnu[index],f"{truth} GNU -> {gnu_pred}")
            _show(axes[1,index],styled[index],f"{truth} styled -> {styled_pred}")
        figure.tight_layout(); figure.savefig(root/f"sample_{sample:02d}_characters.png",dpi=150); plt.close(figure)

        target=batch.get("image",batch.get("word"))[sample]; references=batch["references"][sample]
        figure,axes=plt.subplots(3,1,figsize=(12,6))
        _show(axes[0],references[0],"reference")
        _show(axes[1],target,"real target")
        _show(axes[2],output["word"][sample,...,:length*16],"final WriteViT output")
        figure.tight_layout(); figure.savefig(root/f"sample_{sample:02d}_word.png",dpi=150); plt.close(figure)

        figure,axes=plt.subplots(6,1,figsize=(12,12))
        descriptions=(
            ("scale1","scale 1: 2 x L"),
            ("scale2","scale 2: 4 x 2L"),
            ("scale3","scale 3: 8 x 4L"),
            ("scale4","scale 4: 16 x 8L"),
            ("fused","toRGB fusion + Gaussian feature noise"),
        )
        for axis,(key,title) in zip(axes[:5],descriptions):
            feature=output[key][sample]
            visible_width=max(1,round(feature.shape[-1]*length/full_length))
            _show_feature(axis,feature,title,visible_width)
        _show(axes[5],output["word"][sample,...,:length*16],"CNN-decoded final word")
        figure.tight_layout(); figure.savefig(root/f"sample_{sample:02d}_scales.png",dpi=150); plt.close(figure)
