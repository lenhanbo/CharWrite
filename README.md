# CharWriteViT

Character-first handwriting generation with a WriteViT-style backbone.

```text
writer references --------------------> independent WriteViT Writer Encoder
        |
        +--> independent character-style Writer Encoder
                            |
GNU Unifont tokens -> positional character Transformer
                            |
              characters self-attend + style cross-attend
                            |
          styled glyph preview + refined character tokens
                            |
                 replace GNU query in WriteViT
                            |
       original CrossBlocks -> [styled; refined] grid
                            |
       original multi-scale Transformer/CPE/toRGB
                            |
       original N(0,1) noise + ResBlock/CNN decoder
                            |
                 final word [B,1,32,16L]
```

The only generator change from WriteViT is the character-refinement module
inserted between GNU Unifont and the released WriteViT generator. Its style
encoder is independent from WriteViT's Writer Encoder. There is no merger or
post-generator correction in the baseline.

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
those splits are not true writer-disjoint splits. The joint character critic
does not need IAM character boxes: every decoded cell is aligned one-to-one with
its query token. On critic steps it learns from fixed GNU glyphs plus detached
styled glyphs; on generator steps it is frozen and sends styled-character CE
back through the decoder/refiner. EMNIST/font training remains an optional
standalone initialization ablation only.

The bundled `assets/unifont.pickle` is the GNU Unifont table from WriteViT. Its
frozen 16x16 raster embeddings form the content-token row in the initial
`[D,2,L]` feature grid.

## Install

```bash
git clone <your-repository-url>
cd CharWriteViT
pip install -e .
```

## Joint training on Kaggle

Optional classifier-pretraining ablations can attach a handwriting-font dataset:

```text
handwriting-fonts/
`-- font-family/
    |-- FontName-Regular.ttf
    `-- OFL.txt
```

For standard joint training, check the IAM paths in `configs/kaggle.yaml`:

```yaml
data:
  iam_root: /kaggle/input/iam-dataset/iam_dataset
  prepared_dir: /kaggle/working/charwritevit-data/iam
```

Prepare IAM:

```bash
python -m charwritevit.cli.prepare_iam --config configs/kaggle.yaml
```

Run a short joint smoke test:

```bash
python -m charwritevit.cli.train_iam \
  --config configs/kaggle.yaml \
  --output outputs/iam-smoke \
  --epochs 1 \
  --max-steps 2
```

Then run the complete joint stage:

```bash
python -m charwritevit.cli.train_iam \
  --config configs/kaggle.yaml \
  --output outputs/iam
```

The character classifier is trained on query-labelled GNU and detached styled
glyph cells every critic batch. During generator updates it is frozen, while its styled-glyph CE still
backpropagates through the character decoder and refiner. This mirrors the
real/fake training roles of WriteViT's OCR and Writer ID critics. Resume all
networks together with:

```bash
python -m charwritevit.cli.train_iam \
  --config configs/kaggle.yaml \
  --output outputs/iam \
  --resume outputs/iam/latest.pt
```

The OCR recognizer follows WriteViT's joint-critic schedule. Adam uses
`betas=(0, 0.999)`, no weight decay, and learning rate `5e-5`;
D/OCR/Writer/Character are updated every batch and G every second batch. There
is no GAN warmup or mandatory pretrained auxiliary checkpoint.

Full IAM training has no paired pixel or edge reconstruction loss. Real IAM
images train the recognizer, writer network, and discriminator; GNU/styled cells
train the character classifier; generated words/glyphs are supervised by
character CE, OCR/CTC, writer identity, and GAN losses.

## Evaluation (including FID/KID)

```bash
python -m charwritevit.cli.evaluate \
  --config configs/kaggle.yaml \
  --checkpoint outputs/iam/latest.pt \
  --output outputs/iam/evaluation \
  --max-batches 100 \
  --kid-subset-size 100
```

Evaluation pads all real/generated words to the configured maximum width,
converts grayscale to RGB `[0,1]`, and accumulates standard Inception-2048 FID
and KID alongside CER, word accuracy, SSIM, styled-character accuracy, and style
retrieval. Results are written to `evaluation/metrics.json`. The first run may
download and cache Inception weights; on an offline Kaggle run attach the weight
file and pass `--inception-weights /kaggle/input/.../weights.pth`.

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

Sampling writes each GNU/styled glyph plus `coarse.png` and `word.png`.
Training previews plot losses over time and save character-level and word-level
images using ten fixed validation samples and a fixed noise seed. Every preview
epoch writes `sample_XX_characters.png` (GNU versus decoded styled glyphs),
`sample_XX_scales.png` (Scale 1-4 energy, toRGB/noise fusion, final decode), and
`sample_XX_word.png` (reference, real target, final output). Multi-channel scale
features are rendered as normalized mean absolute channel energy rather than as
fake grayscale word images.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
"# CharWrite" 
