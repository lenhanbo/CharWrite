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
|-- ascii/forms.txt       # preferred form id -> real writer id
|-- linux_gt.txt          # or train_gt.txt + val_gt.txt
`-- words/...
```

The Kaggle preset targets `ngkinwang/iam-dataset`, mounted at
`/kaggle/input/iam-dataset/iam_dataset`. It already provides `linux_gt.txt`,
`train_gt.txt`, `val_gt.txt`, and word images. Without separate writer metadata,
its `a01-000u`-style word folders are treated as pseudo-writers.

The IAM preparer prefers `forms.txt` or XML writer metadata. It can also recover
the mapping from a companion `writer_folder/form_id.png` tree (including a ZIP),
such as the output of the Kaggle forms-dataset creator. As an explicit temporary
fallback, `iam_folder_as_writer: true` treats each word folder as a pseudo-writer;
those splits are not true writer-disjoint splits. The independent stage-1
character classifier does not read IAM:
its required supervision comes from EMNIST ByClass. Optional punctuation
samples can be rendered from `.ttf`/`.otf` handwriting fonts. Every font must have a nearby
`OFL.txt`, `LICENSE.txt`, `LICENSE`, or `UFL.txt` file.
When fonts are available, the classifier trains both sources together using a
configurable 75% EMNIST / 25% synthetic-punctuation epoch mix.

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

The controlled character-ablation path trains directly on IAM. GNU Unifont is
the same frozen content prior as WriteViT; the only experimental addition is
the explicit character branch and its auxiliary losses.

```bash
python -m charwritevit.cli.train_iam \
  --config configs/kaggle.yaml \
  --classifier outputs/classifier/best.pt \
  --warmup-epochs 5 \
  --output outputs/iam
```

The OCR recognizer follows WriteViT's joint-critic schedule: it learns from
real IAM words, is temporarily frozen for fake-word CTC generator updates, and
is stored in the same full checkpoint. `--ocr` remains optional when an OCR
initialization checkpoint is available.

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
