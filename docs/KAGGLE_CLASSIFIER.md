# Character-classifier pretraining on Kaggle

This stage is required before `train_iam`. IAM training loads this checkpoint,
freezes the classifier, and sends styled-character CE only into the character
decoder/refiner.

This stage does **not** read IAM. Attach `crawford/emnist`. EMNIST characters
are resized with aspect-ratio-preserving contain + white padding, never cropped
to the 16x32 classifier canvas. The exact GNU glyph set is repeated as a fixed
anchor for every configured character. A licensed handwriting-font dataset is
still recommended for handwritten punctuation. IAM is needed only by the later
OCR/generator stage.

When fonts are present, training remains a single stage. Each epoch contains
75% EMNIST alphanumeric samples and 25% font-rendered punctuation samples by
default. The font source does not duplicate letters or digits.

## 1. Kaggle setup

Enable a GPU accelerator and attach the Kaggle Dataset `crawford/emnist`. The
loader reads its ByClass CSV directly and writes only a small random-access
offset index under `/kaggle/working`; it does not load the 6 GB dataset into RAM.
Attach a second Dataset of licensed handwriting fonts:

```text
/kaggle/input/handwriting-fonts/
`-- caveat/
    |-- Caveat-Regular.ttf
    `-- OFL.txt
```

Clone and install:

```bash
git clone https://github.com/YOUR_USER/YOUR_REPO.git /kaggle/working/CharWriteViT
cd /kaggle/working/CharWriteViT
pip install -q -e .
```

Update `data.fonts_dir` in `configs/kaggle.yaml` if the attached Dataset slug is
different.

## 2. Prepare sources

```bash
python -m charwritevit.cli.prepare_char_data \
  --config configs/kaggle.yaml
```

Expected output reports a non-zero EMNIST sample count. A zero font count is
valid for EMNIST-only training. If fonts were attached intentionally, check
that every font family directory includes its license.

## 3. Smoke test

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier-v2-smoke \
  --epochs 1 \
  --max-steps 20
```

Confirm these files exist:

```text
latest.pt
best.pt
metrics.jsonl
classifier_curves.png
per_class_accuracy.json
preview/epoch_000_validation.png
preview/epoch_000_gnu.png
```

The smoke accuracy is not meaningful; this run validates data, AMP, checkpoint,
plotting, and previews.

## 4. Full classifier training

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier-v2
```

The default classifier settings are independent from the larger generator:

```yaml
classifier_epochs: 30
classifier_batch_size: 256
classifier_accumulation_steps: 1
classifier_learning_rate: 0.001
classifier_min_learning_rate: 0.00001
classifier_warmup_epochs: 1.0
classifier_unifont_repeats: 256
```

The classifier uses one epoch of linear warmup followed by cosine decay to the
minimum learning rate. Scheduler state is checkpointed. Resume with the same
total `--epochs` chosen for the original run; changing it would redefine the
cosine curve.

`best.pt` is selected by the mean of macro validation accuracy and exact GNU
accuracy. `classifier_curves.png` plots both. Every epoch also saves a normal
validation preview and an all-character GNU preview with true label,
prediction, and confidence.

## 5. Resume and save

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier-v2 \
  --resume outputs/classifier-v2/latest.pt
```

Before closing Kaggle, use **Save Version** or archive the output:

```bash
tar -czf /kaggle/working/charwritevit-classifier-v2.tgz outputs/classifier-v2
```

Use `outputs/classifier-v2/best.pt` as the frozen character evaluator during the
IAM character-ablation experiment.

Stop the notebook here. Do not run `prepare_iam`, `train_synthetic`,
`train_ocr`, or `train_iam` in this classifier stage. IAM/OCR and end-to-end
word training are intentionally handled separately.
