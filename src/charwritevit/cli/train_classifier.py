from __future__ import annotations

from pathlib import Path
import json
import string
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from charwritevit.cli.common import load_vocab, training_parser
from charwritevit.config import load_config
from charwritevit.data.emnist import build_emnist_dataset
from charwritevit.data.synthetic import SyntheticCharacterDataset, discover_fonts
from charwritevit.data.unifont import UnifontCharacterDataset, load_unifont_table
from charwritevit.models.auxiliary import CharacterClassifier
from charwritevit.train.preview import MetricHistory, save_classifier_preview
from charwritevit.train.utils import amp_context, make_scaler, make_warmup_cosine_scheduler, optimizer_step, resolve_device, save_checkpoint, seed_everything


def main():
    args=training_parser("Train the anchored character classifier").parse_args(); cfg=load_config(args.config); seed_everything(int(cfg.get("seed",42)))
    device=resolve_device(); vocab=load_vocab(cfg); data=cfg["data"]; tc=cfg["train"]
    fonts_optional=bool(data.get("classifier_fonts_optional",True))
    fonts=discover_fonts(data["fonts_dir"],bool(data.get("require_font_license",True)),allow_empty=fonts_optional)
    if not fonts:
        warnings.warn("No licensed fonts found. Punctuation is anchored by GNU only, without handwriting-font augmentation.")
    char_root=Path(data.get("char_prepared_dir","data/characters"))
    cache=Path(data.get("emnist_cache",char_root.parent/"emnist-index"))
    auto_download=bool(data.get("auto_download_emnist",True))
    emnist_train=build_emnist_dataset(data["emnist_root"],vocab,train=True,download=auto_download,cache_dir=cache)
    unifont_path=Path(cfg["model"]["unifont_path"])
    if not unifont_path.is_file():
        raise FileNotFoundError(f"Classifier pretraining requires the GNU glyph table: {unifont_path}")
    unifont_table=load_unifont_table(unifont_path,vocab)
    unifont_repeats=int(tc.get("classifier_unifont_repeats",256))
    unifont_train=UnifontCharacterDataset(
        unifont_table,vocab,unifont_repeats,int(data.get("char_width",16))
    )
    unifont_validation=UnifontCharacterDataset(
        unifont_table,vocab,1,int(data.get("char_width",16))
    )
    datasets=[emnist_train,unifont_train]
    punctuation=[char for char in vocab.tokens[3:] if char not in string.ascii_letters+string.digits]
    synthetic_fraction=float(tc.get("classifier_synthetic_fraction",0.25))
    if not 0.0 <= synthetic_fraction < 1.0: raise ValueError("classifier_synthetic_fraction must be in [0, 1)")
    synthetic_samples=0
    if fonts and punctuation and synthetic_fraction>0:
        synthetic_samples=round(len(emnist_train)*synthetic_fraction/(1.0-synthetic_fraction))
        datasets.append(SyntheticCharacterDataset(fonts,vocab,samples=synthetic_samples,characters=punctuation))
    classifier_batch=int(tc.get("classifier_batch_size",tc["batch_size"]))
    loader=DataLoader(ConcatDataset(datasets),batch_size=classifier_batch,shuffle=True,num_workers=int(tc.get("num_workers",2)),pin_memory=True)
    validation_sets=[build_emnist_dataset(data["emnist_root"],vocab,train=False,download=auto_download,cache_dir=cache)]
    if fonts and punctuation and synthetic_fraction>0:
        validation_sets.append(SyntheticCharacterDataset(fonts,vocab,samples=int(tc.get("classifier_synthetic_val_samples",10000)),characters=punctuation))
    validation=ConcatDataset(validation_sets)
    validation_loader=DataLoader(validation,batch_size=classifier_batch,shuffle=False,num_workers=int(tc.get("num_workers",2)),pin_memory=True)
    epochs=args.epochs or int(tc.get("classifier_epochs",tc["epochs"])); accumulation=int(tc.get("classifier_accumulation_steps",1))
    model=CharacterClassifier(len(vocab)).to(device); base_lr=float(tc.get("classifier_learning_rate",tc["learning_rate"])); opt=torch.optim.AdamW(model.parameters(),lr=base_lr); scaler=make_scaler(device,tc.get("amp",True))
    updates_per_epoch=max(1,len(loader)//accumulation); total_updates=max(1,updates_per_epoch*epochs)
    warmup_updates=min(total_updates-1,round(updates_per_epoch*float(tc.get("classifier_warmup_epochs",1.0))))
    min_lr=float(tc.get("classifier_min_learning_rate",1e-5)); scheduler=make_warmup_cosine_scheduler(opt,total_updates,warmup_updates,min_lr/base_lr)
    step=0; start_epoch=0; best_accuracy=0.0; history=MetricHistory(args.output)
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if int(state.get("classifier_data_version",0)) != 2:
            raise ValueError("This checkpoint predates uncropped EMNIST + GNU anchoring; retrain the classifier from scratch")
        model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); scaler.load_state_dict(state.get("scaler",{})); step=int(state.get("step",0)); start_epoch=int(state.get("epoch",-1))+1; best_accuracy=float(state.get("best_accuracy",0.0))
        if "scheduler" not in state: raise ValueError("Resume checkpoint predates the LR scheduler; restart classifier training so the schedule is well-defined")
        if int(state.get("scheduler_total_updates",-1)) != total_updates: raise ValueError("Resume must use the same total --epochs as the original scheduler run")
        scheduler.load_state_dict(state["scheduler"])
    for epoch in range(start_epoch, epochs):
        model.train(); train_loss=train_correct=train_count=0
        for images,labels in loader:
            images,labels=images.to(device),labels.to(device)
            with amp_context(device,tc.get("amp",True)): logits=model(images); loss=F.cross_entropy(logits,labels)
            with torch.no_grad(): train_loss+=float(loss.detach())*labels.numel(); train_correct+=(logits.argmax(-1)==labels).sum().item(); train_count+=labels.numel()
            if optimizer_step(loss,opt,scaler,accumulation,step): scheduler.step()
            step+=1
            if args.max_steps and step>=args.max_steps: break
        model.eval(); val_loss=val_correct=val_count=0; preview=None; class_correct=torch.zeros(len(vocab),dtype=torch.long); class_count=torch.zeros(len(vocab),dtype=torch.long)
        with torch.inference_mode():
            for batch_index,(images,labels) in enumerate(validation_loader):
                images,labels=images.to(device),labels.to(device); logits=model(images); batch_loss=F.cross_entropy(logits,labels)
                val_loss+=float(batch_loss)*labels.numel(); val_correct+=(logits.argmax(-1)==labels).sum().item(); val_count+=labels.numel()
                predictions=logits.argmax(-1); cpu_labels=labels.cpu(); cpu_correct=predictions.eq(labels).cpu()
                class_count.scatter_add_(0,cpu_labels,torch.ones_like(cpu_labels)); class_correct.scatter_add_(0,cpu_labels,cpu_correct.long())
                if preview is None: preview=(images.cpu(),labels.cpu(),logits.cpu())
                if args.max_steps and batch_index>=9: break
        gnu_loader=DataLoader(unifont_validation,batch_size=len(unifont_validation),shuffle=False,num_workers=0)
        gnu_images,gnu_labels=next(iter(gnu_loader)); gnu_images,gnu_labels=gnu_images.to(device),gnu_labels.to(device)
        with torch.inference_mode():
            gnu_logits=model(gnu_images); gnu_loss=F.cross_entropy(gnu_logits,gnu_labels)
            gnu_accuracy=float(gnu_logits.argmax(-1).eq(gnu_labels).float().mean())
        observed=class_count.gt(0); per_class=class_correct.float()/class_count.clamp_min(1); macro=float(per_class[observed].mean()) if observed.any() else 0.0
        classifier_score=0.5*(macro+gnu_accuracy)
        metrics={"epoch":epoch,"step":step,"train_loss":train_loss/max(1,train_count),"train_accuracy":train_correct/max(1,train_count),"val_loss":val_loss/max(1,val_count),"val_accuracy":val_correct/max(1,val_count),"val_macro_accuracy":macro,"gnu_loss":float(gnu_loss),"gnu_accuracy":gnu_accuracy,"classifier_score":classifier_score,"learning_rate":opt.param_groups[0]["lr"],"synthetic_samples":synthetic_samples,"unifont_samples":len(unifont_train)}
        history.append(metrics); history.plot("classifier_curves.png")
        (Path(args.output)/"per_class_accuracy.json").write_text(json.dumps({vocab.tokens[index]:float(per_class[index]) for index in range(len(vocab)) if observed[index]},ensure_ascii=False,indent=2),encoding="utf-8")
        if preview is not None:
            save_classifier_preview(*preview,vocab.tokens,Path(args.output)/"preview"/f"epoch_{epoch:03d}_validation.png",int(tc.get("preview_samples",16)),title="EMNIST / synthetic validation")
        save_classifier_preview(gnu_images.cpu(),gnu_labels.cpu(),gnu_logits.cpu(),vocab.tokens,Path(args.output)/"preview"/f"epoch_{epoch:03d}_gnu.png",len(unifont_validation),columns=10,title=f"GNU anchor accuracy: {gnu_accuracy:.1%}")
        score=classifier_score; is_best=score>=best_accuracy; best_accuracy=max(best_accuracy,score)
        checkpoint=dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),scheduler_total_updates=total_updates,scaler=scaler.state_dict(),classifier_data_version=2,epoch=epoch,step=step,best_accuracy=best_accuracy,config=cfg)
        save_checkpoint(Path(args.output)/"latest.pt",**checkpoint)
        if is_best: save_checkpoint(Path(args.output)/"best.pt",**checkpoint)
        print(metrics)
        if args.max_steps and step>=args.max_steps: break


if __name__ == "__main__": main()

