# CharWriteViT

Character-first handwriting generation with a WriteViT-style backbone.

```text
text + writer references
        |
        +--> ViT writer encoder
        |
GNU Unifont content tokens
        |
        +--> per-character glyph images --> character classifier loss
        |              |
        |              +--> bidirectional character contextualizer
        |                              |
        +------------------------------+
                       |
             [context; content]
                       |
                 [B, D, 2, L]
                       |
       multi-scale Transformer + CPE
                       |
          toRGB multi-scale fusion
                       |
        noise + ResBlock/CNN decoder
                       |
             final word [B,1,32,16L]
```

The merger is intentionally disabled for the baseline. It will only be added
after the multi-scale generator has been trained and inspected. See
`IMPLEMENTATION_PLAN.md` for the persistent implementation brief.

## Data

IAM is not redistributed. The IAM root must contain:

```text
iam_dataset/
|-- ascii/forms.txt       # form id -> real writer id (required)
|-- linux_gt.txt          # or train_gt.txt + val_gt.txt
`-- words/...
```

The IAM preparer creates deterministic writer-disjoint train/validation/test
manifests. The independent stage-1 character classifier does not read IAM:
its supervision comes from EMNIST ByClass and synthetic samples
rendered from `.ttf`/`.otf` handwriting fonts. Every font must have a nearby
`OFL.txt`, `LICENSE.txt`, `LICENSE`, or `UFL.txt` file.

The bundled `assets/unifont.pickle` is the GNU Unifont table from WriteViT. Its
frozen 16x16 raster embeddings form the content-token row in the initial
`[D,2,L]` feature grid.

## Install

```bash
git clone <your-repository-url>
cd CharWriteViT
pip install -e .
```

## First stage on Kaggle: character classifier

The expanded step-by-step guide is in `docs/KAGGLE_CLASSIFIER.md`.

Attach a handwriting-font dataset with this structure:

```text
handwriting-fonts/
`-- font-family/
    |-- FontName-Regular.ttf
    `-- OFL.txt
```

Check these paths in `configs/kaggle.yaml`:

```yaml
data:
  emnist_root: /kaggle/input/emnist
  emnist_cache: /kaggle/working/charwritevit-data/emnist-index
  fonts_dir: /kaggle/input/handwriting-fonts
  char_prepared_dir: /kaggle/working/charwritevit-data/characters
  prepared_dir: /kaggle/working/charwritevit-data/iam
```

Validate EMNIST and discover licensed fonts (IAM is not used by this command):

```bash
python -m charwritevit.cli.prepare_char_data \
  --config configs/kaggle.yaml
```

Run a short smoke test first:

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier-smoke \
  --epochs 1 \
  --max-steps 20
```

Then run the complete classifier stage:

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier
```

The stage writes:

```text
outputs/classifier/
|-- latest.pt
|-- best.pt
|-- metrics.jsonl
|-- classifier_curves.png
`-- preview/epoch_XXX.png
```

Each preview shows the input character, true label, predicted label, and model
confidence. Resume an interrupted Kaggle session with:

```bash
python -m charwritevit.cli.train_classifier \
  --config configs/kaggle.yaml \
  --output outputs/classifier \
  --resume outputs/classifier/latest.pt
```

Archive `outputs/classifier` as a Kaggle notebook output or Kaggle Dataset before
ending the session.

## Remaining stages

```bash
python -m charwritevit.cli.train_synthetic \
  --config configs/kaggle.yaml \
  --classifier outputs/classifier/best.pt \
  --output outputs/synthetic

python -m charwritevit.cli.train_ocr \
  --config configs/kaggle.yaml \
  --output outputs/ocr

python -m charwritevit.cli.train_iam \
  --config configs/kaggle.yaml \
  --classifier outputs/classifier/best.pt \
  --ocr outputs/ocr/latest.pt \
  --synthetic-checkpoint outputs/synthetic/latest.pt \
  --warmup-epochs 5 \
  --output outputs/iam
```

Edge reconstruction is logged as a diagnostic but its training weight is zero
by default. It should only be enabled after inspecting baseline results.

## Sampling

```bash
python -m charwritevit.cli.sample \
  --config configs/kaggle.yaml \
  --checkpoint outputs/iam/latest.pt \
  --text minimum \
  --reference /path/to/ref1.png \
  --reference /path/to/ref2.png \
  --output outputs/sample
```

Sampling writes each raw/contextual glyph plus `coarse.png` and `word.png`.
Training previews plot losses over time and save character-level and word-level
images using fixed validation samples and noise seeds.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
"# CharWrite" 
