from __future__ import annotations

import argparse
from pathlib import Path

import torch

from charwritevit.cli.common import build_generator, load_vocab
from charwritevit.config import load_config
from charwritevit.data.image import canonical_word, tensor_to_pil
from charwritevit.train.utils import load_checkpoint, resolve_device


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--checkpoint",required=True); parser.add_argument("--text",required=True); parser.add_argument("--reference",action="append",required=True); parser.add_argument("--output",required=True)
    args=parser.parse_args(); cfg=load_config(args.config); device=resolve_device(); vocab=load_vocab(cfg); dc=cfg["data"]
    if not args.text or len(args.text)>int(dc.get("max_word_length",24)): raise ValueError("Text length is outside configured range")
    model=build_generator(cfg,vocab).to(device); state=load_checkpoint(args.checkpoint,device); model.load_state_dict(state.get("generator",state.get("model"))); model.eval()
    from PIL import Image
    refs=[]
    for path in args.reference:
        with Image.open(path) as image: refs.append(canonical_word(image.convert("L"),min(max(len(args.text),5),24),32,16))
    wanted=int(dc.get("style_references",4)); refs=(refs*((wanted+len(refs)-1)//len(refs)))[:wanted]
    references=torch.stack(refs).unsqueeze(0).to(device); ids=torch.tensor([vocab.encode(args.text)],device=device)
    with torch.inference_mode(): out=model(ids,references)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    for i,char in enumerate(args.text):
        tensor_to_pil(out["raw_glyphs"][0,i]).save(root/f"{i:02d}_{ord(char):04x}_raw.png")
        tensor_to_pil(out["contextual_glyphs"][0,i]).save(root/f"{i:02d}_{ord(char):04x}_context.png")
    tensor_to_pil(out["coarse_word"][0]).save(root/"coarse.png"); tensor_to_pil(out["word"][0]).save(root/"word.png")
    print(root/"word.png")


if __name__ == "__main__": main()

