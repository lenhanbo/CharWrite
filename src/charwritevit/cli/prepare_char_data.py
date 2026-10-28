from __future__ import annotations

import argparse
import json
from pathlib import Path

from charwritevit.config import load_config
from charwritevit.data.emnist import build_emnist_dataset
from charwritevit.data.synthetic import discover_fonts, font_license
from charwritevit.data.vocab import CharacterVocabulary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--download-emnist", action="store_true")
    args=parser.parse_args(); cfg=load_config(args.config); data=cfg["data"]
    vocab=CharacterVocabulary(cfg["model"]["alphabet"])
    char_root=Path(data.get("char_prepared_dir","data/characters"))
    cache=Path(data.get("emnist_cache",char_root.parent/"emnist-index"))
    emnist=build_emnist_dataset(data["emnist_root"],vocab,download=args.download_emnist,cache_dir=cache)
    fonts=discover_fonts(data["fonts_dir"],bool(data.get("require_font_license",True)))
    font_root=Path(data["fonts_dir"])
    manifest={"emnist_samples":len(emnist),"fonts":[{"path":str(x),"license":str(font_license(x,font_root) or "unverified")} for x in fonts],"alphabet":cfg["model"]["alphabet"]}
    # Character classification is an independent first stage.  Keep its
    # manifest separate from the later IAM preparation output.
    output=char_root; output.mkdir(parents=True,exist_ok=True)
    (output/"character_sources.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"emnist_samples":len(emnist),"font_count":len(fonts)},indent=2))


if __name__ == "__main__": main()

