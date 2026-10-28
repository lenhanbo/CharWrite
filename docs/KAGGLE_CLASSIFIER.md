# Train the character classifier on Kaggle

This is the first training stage. Do not start synthetic-word or IAM training
until `best.pt`, the curves, per-class accuracy, and preview images have been
inspected.

This stage does **not** read IAM. Attach `crawford/emnist`. A licensed
handwriting-font dataset is optional: without it the classifier trains on the
62 EMNIST alphanumeric classes, while punctuation remains untrained. IAM is
needed by later OCR and generator stages.

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
  --output outputs/classifier-smoke \
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
preview/epoch_000.png
```

The smoke accuracy is not meaningful; this run validates data, AMP, checkpoint,
plotting, and previews.

## 4. Full classifier training

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier
```

The default classifier settings are independent from the larger generator:

```yaml
classifier_epochs: 10
classifier_batch_size: 256
classifier_accumulation_steps: 1
classifier_learning_rate: 0.001
```

`best.pt` is selected by macro validation accuracy rather than overall accuracy,
so rare punctuation cannot be hidden by the much larger EMNIST alphanumeric
classes.

## 5. Resume and save

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier \
  --resume outputs/classifier/latest.pt
```

Before closing Kaggle, use **Save Version** or archive the output:

```bash
tar -czf /kaggle/working/charwritevit-classifier.tgz outputs/classifier
```

Use `outputs/classifier/best.pt` for the next synthetic-glyph stage.
