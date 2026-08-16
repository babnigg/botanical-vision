# Botanical Vision

Fine-grained flowering-plant classification for Advanced Computer Vision (ADSP 32023) —
a gardener's tool with a computer-vision spine: **Identify** a plant from a photo,
collect species in a **Toolbox**, and **Compose** a garden design.

The current stage builds and improves the classifier: photos of flowering plants labeled
by species, sourced from citizen-science observations on iNaturalist.

## What we're actually doing (start here)

This is a **computer-vision class**. The project is **two pillars**:

1. **The classification pipeline** — the fine-grained classifier plus everything
   attached to it: the detection experiment (06), the segmentation experiment (07),
   and the planned **Stylize** layer (08/09: classify → retrieve conspecifics →
   generate new art *of that species*, beyond simple style transfer).
2. **The landscape generator** — Compose (10–16): symbolic planting-plan generation,
   decode-time search, ControlNet rendering, real-plan harvesting (pixel labeling +
   structured permapeople plans), a literature-grounded metric suite, and the
   curated design-layer reframe with blinded evaluation.

Negative results are kept and reported (e.g., subject-cropping hurts top-1 — measured);
the course rewards documented iteration, not just wins.

For the classifier pillar, the loop is three steps, and only the middle one is really
yours:

1. **Get the data** — one line; it's already on Hugging Face, you never scrape anything.
2. **Improve the model** — iterate in `notebooks/04_train_improved.ipynb`.
3. **Share your best model** so the team can compare — two commands:

   ```bash
   python -m share.publish --checkpoint checkpoints/resnet50_improved_best.pt --name my-model
   python -m share.leaderboard        # ranks everyone's shared models on the same test set
   ```

That's the entire job. Publishing uploads your weights to a shared Hugging Face model
repo (not git); the leaderboard pulls everyone's and scores them identically, so "whose
model is best" is an objective number. (`python -m share.score --checkpoint <ckpt>` checks
your own first.)

> There's also a separate **demo app** in `demo/` (Identify · Toolbox · Compose) that shows
> the classifier as a product. It's self-contained and optional — you never need it to
> train, share, or compare models. See [`demo/README.md`](demo/README.md).

## Data

Every image is a research-grade (community-verified) iNaturalist observation of a
flowering plant. We scope the dataset to species that are well-represented, so each
class has enough images to train on.

- **Which species:** flowering-plant species (Magnoliopsida + Liliopsida) with at
  least 2,000 research-grade observations — 4,172 selected, **4,094 in the final
  dataset** after dropping names that don't resolve on iNaturalist or lack enough
  unique images.
- **How many images:** up to 100 photos per species; **~408K after removing exact
  duplicates** (285K train / 61K val / 61K test, stratified within each species).
- **Where it lives:** images are published to Hugging Face, not committed to git, so
  the group loads one canonical copy instead of re-scraping.

### Source & citation

The list of which species exist and how often they're observed comes from a GBIF
occurrence download (iNaturalist Research-grade Observations, filtered to flowering
plants with still images). The photos themselves are pulled from the iNaturalist API.

> GBIF.org (8 July 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.3hragg

Individual photos retain their own iNaturalist licenses and creator attribution.

### How the data was built

```
GBIF species list  ─►  01_eda_species.ipynb  ─►  selected_species.csv  ─►  download_inaturalist.py  ─►  images/  ─►  02_eda_images.ipynb  ─►  splits.csv
 (which species,          (pick species by            (target list)          (pull photos from iNat)                     (inventory, dedup,
  how observed)            observation count)                                                                              train/val/test split)
```

---

## Technical

Runs on Windows and macOS with Python 3.10+. All paths are handled portably, so a
fresh clone works on either OS. (On Windows, `load_dataset` may print a harmless
symlink caching warning.)

### Using the dataset (everyone)

This is all a teammate needs. You do **not** need to run the GBIF download or the
scraper — the dataset is already published to Hugging Face.

```bash
pip install -r requirements.txt
```

```python
from datasets import load_dataset
ds = load_dataset("dbabnigg/botanical-vision-256")   # train / val / test
```

The dataset is public, so no Hugging Face account or login is required. Use the
downscaled **`botanical-vision-256`** (~9 GB, long edge ≤ 256px) — it downloads in
minutes, fits Colab's disk, and is **exactly what the training notebooks load**, so
teammates and the notebooks read the same build. A full-resolution
`dbabnigg/botanical-vision` (~25 GB, identical schema) also exists for archival, but
you don't need it to train or evaluate.

**Torch / GPU.** `requirements.txt` installs a CPU build of PyTorch, which is enough
to run everything (slowly). For GPU training, install the CUDA build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Note: PyTorch must be **≥ 2.3** to work with NumPy 2. The notebooks pick GPU
automatically when available and fall back to CPU otherwise; Google Colab is a free
GPU option.

### Which notebooks do I run?

Classifier work is **03 → 05**, segmentation is **07**, and the Compose notebooks
**10 → 12** are optional — all of them auto-load what they need (dataset from Hugging
Face, plans generated procedurally), so you never need local data.

| Notebook | Who | What to do |
| --- | --- | --- |
| `01_eda_species` | maintainer only | **don't re-run** — rebuilds the species selection; read only |
| `02_eda_images` | maintainer only | **don't re-run** — inventories raw images + writes the split; read only |
| `03_train_classifier` | everyone | **run-only** baseline (fixed reference) — don't experiment here |
| `04_train_improved` | everyone | **where classifier work goes** — all classifier improvements live here |
| `04b_train_vit` | optional | ViT-base challenger (HF `transformers`) — second champion-challenger pair for problem 1 |
| `05_evaluate` | everyone | metrics/visuals on a trained checkpoint |
| `06_detect_subject` | optional | zero-shot YOLO subject localizer + Δtop-1 (see below) |
| `07_train_segmentation` | everyone | distilled plant/background segmenter — feeds the classifier as a soft-mask preprocessor |
| `08_style_baseline` | placeholder | Stylize baseline — neural style transfer (scope only) |
| `09_style_retrieval` | placeholder | Stylize enhanced — classify → retrieve → generate artwork (scope only) |
| `10_garden_layout` | optional | Compose — rule-generated planting plans + autoregressive layout transformer |
| `11_complete_layout` | optional | Compose — masked-diffusion completion (pin toolbox plants, infill the rest; sun-conditioned) |
| `12_render_plan` | optional | Compose — seg-ControlNet render of any plan (zero training) |
| `13_refine_layout` | optional | Compose — designed corpus, count-token conditioning, best-of-N + repair decoding |
| `14_real_plans` | optional | Compose — label real drawings (Hough+edge verification) → transfer pretraining |
| `15_layered_layout` | optional | Compose — layered corpus (strata/drift/rhythm), literature metrics, real-plan calibration |
| `16_curated_design` | optional | Compose — the reckoning: metric circularity exposed, design-layer reframe, realism metric, blinded pairs |

Training writes a `.pt` to `checkpoints/`. To share it with the team, run
`python -m share.publish` (see *start here* above).

### Train a classifier

Two notebooks, both fine-tuning ResNet-50 on the species splits (`data/splits.csv`)
and reporting top-1 / top-5 accuracy:

- `notebooks/03_train_classifier.ipynb` — the **baseline**: a plain fine-tune, kept
  fixed as the comparison point (run it, don't modify it). Reference run on all
  4,094 species (Kaggle T4, 5 epochs, 2026-08-08): **test top-1 0.549 · top-5 0.755**
  (best val 0.555) — trained headless via `kaggle/` (a 4 GB laptop GPU would take
  ~100 h; the T4 does it in ~3 h).
- `notebooks/04_train_improved.ipynb` — the **active model** where improvements go:
  stronger augmentation (RandAugment, color jitter, random erasing), label
  smoothing, discriminative learning rates, a cosine schedule, more epochs, and
  best-val checkpointing. Reference run on all 4,094 species (Kaggle T4,
  2026-08-09): **test top-1 0.620 · top-5 0.786** (best val 0.630) —
  **+7.1pp top-1** over the fixed baseline under the same full-vocabulary
  protocol.

Both have an `N_SPECIES` toggle — set it to an int (e.g. 100) for a quick run, or
leave it `None` to train on all species (a multi-hour run on the full dataset).

A third architecture, `notebooks/04b_train_vit.ipynb`, fine-tunes **ViT-base**
(`google/vit-base-patch16-224-in21k`) on the same splits — the transformer
challenger to the ResNet champion. Reference run on all 4,094 species (Kaggle
T4, 5 epochs, 2026-08-16): **test top-1 0.604 · top-5 0.804** (best val 0.605) —
above the 5-epoch baseline, below the tuned ResNet at a third of its epochs;
published as `vit-base` on the shared repo.

**Resumable training.** `bv.fit()` saves full state (model, optimizer, scheduler,
epoch, step, best-val, history) every `ckpt_every` steps and writes two checkpoints
per run: `{run_name}_last.pt` (rolling) and `{run_name}_best.pt` (best val). On
restart it auto-resumes from `_last.pt` if its config signature matches the current
setup — even mid-epoch — so a long run survives interruptions. A changed config
(species count, epochs) starts fresh; pass `fresh=True` to force a fresh run. On
**Google Colab**, `bv.checkpoint_dir()` mounts Drive and checkpoints there
automatically, so progress survives disconnects.

`05_evaluate.ipynb` loads `{run_name}_best.pt` — set `CHECKPOINT` there to the model
you want to evaluate.

### Running on Google Colab

The training and eval notebooks run **unchanged on Colab** — they auto-detect the
environment. When the local `../data` files aren't present (i.e. on a Colab runtime),
they load the dataset straight from HuggingFace, checkpoint to mounted Google Drive,
and turn on mixed precision with a larger batch to use the T4's tensor cores (roughly
3-6x faster than a small laptop GPU). Open the notebook via the Colab VS Code
extension or from GitHub on colab.research.google.com, set a T4 runtime, and run.
Because training is resumable, a dropped session just picks up from the last
checkpoint on Drive.

### Evaluate

`notebooks/05_evaluate.ipynb` loads a saved checkpoint (baseline or improved) and
goes beyond top-1/top-5: macro-F1 and balanced accuracy (which weight every species
equally despite the class imbalance), **taxonomy-aware accuracy** (right genus /
right family even when the exact species is wrong), and visuals — per-species
accuracy spread, a family-level confusion matrix, most-confused species pairs,
error rate vs. derivable signals (a greenness/foliage proxy, image count,
resolution), a prediction grid, a t-SNE of the learned embeddings by family, and
Grad-CAM maps of where the model looks.

### Subject localizer (optional)

`notebooks/06_detect_subject.ipynb` addresses the `subject-localizer` row in
`models/planned.json`: it uses a **pretrained, open-vocabulary YOLO (YOLO-World)** to draw
a box around the plant/flower subject in a photo — zero-shot, since the dataset has no
bounding-box labels to fine-tune a detector on. It reports **Δtop-1** (does cropping to the
localized subject change a trained classifier's accuracy?) on a small streamed sample
instead of IoU/mAP, which need ground-truth boxes this dataset doesn't have.

### Segmentation (background masking → Δtop-1)

A distilled plant/background segmenter, originally intended as a preprocessing
stage for the classifier. **The measurement is in, and it's a negative result**
(`scripts/measure_delta_top1.py`, n=3000, full-vocabulary classifier,
2026-08-09): soft-mask preprocessing costs **−31.8pp top-1 / −39.3pp top-5** —
the mask geometry always blurs the crop periphery and under-covering masks blur
plant matter, so the classifier loses signal it trained on. Together with the
detection result (cropping: −4pp), inference-time subject isolation is now
measured twice and rejected twice.

The follow-up — moving background suppression into *training* augmentation
(`04_train_improved`'s `BG_AUG` toggle: blur the background with the student's
mask at p=0.5) — was A/B'd under identical code (100 species, 15 epochs,
2026-08-09): **bg-aug 0.692/0.900 vs control 0.697/0.901 top-1/top-5 — a wash
at subset scale.** A full-vocabulary re-run under the champion recipe is staged:
`scripts/publish_student_masks_full.py` publishes a paired
`{image, mask, species, split}` companion dataset for the HF path, and
`04_train_improved` flips `BG_AUG=True` on top of it. Compare via
`python -m share.leaderboard`, scoring both at `--limit 3000` (the default 300
is too noisy for sub-2pp deltas).

Three experiments, one conclusion so far: backgrounds carry usable signal
(habitat context) for fine-grained species ID; don't fight them.

What ships: the **UX overlay** in the demo's Identify tab — the mask is returned
as a base64 PNG and rendered as a toggleable "what the model attends to" overlay,
while the classifier always sees the original image.

**Approach:** teacher–student distillation. FastSAM generates pseudo-masks over
a stratified 20K subset of the train split; a small student
(DeepLabV3 + MobileNetV3-Large, ~11 MB) is trained on those masks and is what
the demo actually deploys.

**Pipeline:**

```
HF source split  ─►  scripts/generate_pseudo_masks.py  ─►  <you>/botanical-vision-256-masks  ─►  notebooks/07_train_segmentation.ipynb  ─►  checkpoints/deeplabv3_mnv3_seg_best.pt
                    (FastSAM, maintainer only)               (companion HF dataset)               (student, run anywhere; resumable)
```

**Teammates:** open `notebooks/07_train_segmentation.ipynb`, set
`hf_masks_repo` in the `bv.setup(...)` call, and run — the notebook auto-loads
the masks from Hugging Face just like 03/04 do for images.

**Maintainer only (regenerate masks):**

```bash
pip install ultralytics    # FastSAM teacher weights
python scripts/generate_pseudo_masks.py \
    --repo <you>/botanical-vision-256-masks \
    --n-train 20000 --n-val 1000 --n-test 1000
```

**Demo integration.** If `checkpoints/*_seg_best.pt` exists when the demo
backend loads, Identify auto-picks the newest and shows the mask overlay; no
selection UI needed. Nothing there = segmentation is a no-op, classifier runs
unchanged.

### Stylize (scoped, not yet built)

`08_style_baseline` and `09_style_retrieval` scope the **Stylize** module (photo →
artwork). 08 is the floor: plain neural style transfer (VGG-19 Gatys). 09 is the
point: **classify → retrieve → generate** — identify the species with the shared
classifier, retrieve conspecific exemplars from an embedding index, and condition
generation on them so the artwork stays botanically faithful, scored partly by
whether the classifier still recognizes it. Both are markdown scaffolds with todo
cells.

### Garden layout (Compose, optional)

Notebooks 10–12 build the **Compose** module: generate a planting plan (which plants,
where, how many) for a garden bed, then render it. Published generative garden design
works in pixel space (pix2pix/GANs on ~100 scraped plan images); this treats a plan as a
**layout** — (species, x, y, spread) tuples — the way LayoutTransformer/LayoutDM do. No
planting-plan dataset exists, so plans are sampled from horticultural rules (taller in
back, odd-count drifts, spacing from mature spread), and the same rules double as the
scoring metrics. Shared primitives (palette, rule generator, metrics, drawing) live in
`bvtrain/garden.py`; each notebook trains locally in minutes with live progress bars.

- **`10_garden_layout`** — random dummy baseline → rule generator → a ~1M-param
  autoregressive layout transformer with grammar-constrained sampling. Scores: random
  0.63 < transformer 0.76 < rules 0.92. Two structural limits found: no arbitrary
  pinning (prefix-only conditioning) and drifts don't stay odd.
- **`11_complete_layout`** — swaps in **masked discrete diffusion** (absorbing-state,
  MaskGIT-style confidence unmasking): a bidirectional transformer over a fixed plan
  canvas. Enables the real Compose interaction — pin toolbox species anywhere, the model
  infills placement and companions — and adds **sun** as a site condition (learned
  perfectly, 1.00 compliance). Overall 0.78; drift parity remains the open problem.
  Training is checkpointed and resumable (`checkpoints/garden_maskdiff_{last,best}.pt`).
- **`12_render_plan`** — a plan is already a segmentation map, so SD 1.5 +
  `control_v11p_sd15_seg` renders it with zero training (~10 min/image on a 4 GB GPU
  via CPU offload in fp32, ~10 s on a T4 in fp16). Includes the conditioning-scale
  ablation and a CLIP-scored style comparison. Only future style-LoRA /
  custom-ControlNet *training* needs Colab.
- **`13_refine_layout`** — iteration 3, every step A/B-measured: a *designed* teacher
  corpus (elongated drifts, theme repetition, 60/30/10 color budget, bloom set-cover —
  each rule also a metric), per-species **count tokens** for global conditioning, and
  **decode-time search** (best-of-8 reranked by the metrics + constraint repair).
  Result: model 0.800 → 0.891 ≈ teacher 0.900 on `score2`; drift 0.45 → 0.94; CLIP
  aesthetic edge survives the render. Measured caveat: count tokens obey only softly
  (0.72 plants/species), so guaranteed pinning still uses slot tokens.
- **`14_real_plans`** — real drawings as the next teacher: a Hough + edge-verification
  labeler (validated at precision ≈ 1.0 / recall ≈ 0.94–1.0 on known ground truth, with
  a density gate that correctly rejects off-genre engravings) converts circle-symbol
  planting plans into canvases, size-pseudo-labeled. Drop collected drawings into
  `data/real_plans/` (not committed); transfer pretraining arms itself at ≥ 20 plans.
- **`15_layered_layout`** — iteration 4: representation + metrics rebuilt from planting
  design literature before retraining. Plans become **strata** (groundcover carpets the
  bed under taller layers — Rainer & West's layer shares), drifts are elongated,
  oblique, and interlocking (Jekyll) with a theme plant repeated at quasi-regular
  intervals (Oudolf). `score3` is deliberately *not* the generator's rules restated:
  form/texture adjacency (UF/IFAS), min-over-season interest, Moon–Spencer hue harmony,
  per-species clustering (Ripley's K). Calibration on ~330 **structured permapeople.org
  plans** (`scripts/harvest_permapeople.py` → coordinates + species, no pixel
  extraction; `scripts/permapeople_plans.py` parses + scores them): designed ornamental
  beds score 0.67–0.81 vs 0.47 corpus mean, so the suite ranks real design quality —
  the anti-circularity fix 13/14 lacked. Palette 16 → 21 species (appended; old
  checkpoints keep their indices) with `form`/`tex`/`persist` traits + common names.
- **`16_curated_design`** — the reckoning, driven by looking at outputs: 15's
  0.88-scoring plan, redrawn as a designer's patch plan, was a groundcover carpet with
  seven plants (self-authored metrics + argmax search = reward hacking, Gao et al.
  2023). The fix: (1) **genre reframe** — plans become the *design layer* real
  drawings show (~35 symbols, anchors, theme masses, rhythm; `gen_plan4`); measured
  against real-bed statistics, the v3 corpus was indistinguishable from random (3.63
  vs 3.51) while v4 measures 0.81. (2) **honest evaluation** — `realism` (distance
  to real-bed stats we didn't author) is primary; `score3` is held-out report-only;
  final judgment is blinded pairwise sheets. (3) **simplification** — count tokens
  dropped (canvas 354 → 147), carpets dropped, `curate4` editorial pass added.
  Anonymized pairwise eval (2 judges, 12 randomized pairs): **the rule teacher won
  10–2**, so the demo routes by capability — no pins → curated rules, pins →
  the diffusion model (its unique strength: infilling around fixed slots, trimmed
  to the exact pinned counts). Renders are color-grounded: bloom/leaf colors
  measured from dataset photos (`scripts/measure_species_colors.py` →
  `bvtrain/species_colors.json`) seed a fleck sketch that SD1.5 + ControlNet
  img2img turns into a watercolor.

### Rebuilding the dataset (maintainer only)

Everything below regenerates the dataset from scratch and only needs to be run once
by whoever maintains it — teammates can skip this entirely. It also documents exactly
how the published data was produced.

**1. Select species.** Run `notebooks/01_eda_species.ipynb`. Reads the GBIF species
list (`data/raw/gbif_species_list.csv`), explores the observation-count distribution
and taxonomy, and writes `data/selected_species.csv` (the ≥2,000-observation species).

**2. Download images.**

```bash
python scripts/download_inaturalist.py --images_per_species 100 --workers 8
```

Resolves each scientific name to an iNaturalist taxon (GBIF and iNaturalist use
different taxonomy keys), then pulls research-grade photos.

- **Resumable.** Skips species/images already on disk, caches name→taxon lookups in
  `data/inat_taxon_map.json`, and writes each image atomically. Rerun to continue.
- **Runtime.** ~4.5 hours for the full run at 8 workers. Test with
  `--limit 20 --images_per_species 20` first.
- **Rate limits.** API calls throttled under iNaturalist's ~100 req/min; image files
  fetched concurrently from their CDN.

Images land in `data/raw/images/{speciesKey}_{Genus_species}/{speciesKey}_{photoId}.jpg`.

**3. Inspect & split.** Run `notebooks/02_eda_images.ipynb` for per-class counts,
dimensions, duplicate detection (exact and cross-species label leakage), and sample
grids. Writes a stratified 70/15/15 `data/splits.csv`.

**4. Publish to Hugging Face.** Requires a Hugging Face account and a write token.
Publish two builds — full resolution for archival, and a downscaled build for Colab:

```bash
huggingface-cli login
python scripts/upload_to_hf.py --repo <username>/botanical-vision                  # full res (~25 GB)
python scripts/upload_to_hf.py --repo <username>/botanical-vision-256 --max-size 256  # ~9 GB, Colab-friendly
```

Reads `splits.csv` and `selected_species.csv`, builds the train/val/test
`DatasetDict` (each image labeled by species, with genus/family/order/class
metadata), pushes it to the Hub, and writes the dataset card. The `-256` build is
what the training notebooks read. Defaults to public; pass `--private` for a private
dataset. The full dataset is ~25 GB, so the push takes a while — it's **resumable and
retries through brief internet outages**, so it's safe to re-run.

### Layout

```
project/
├── data/
│   ├── raw/gbif_species_list.csv     # GBIF species list (input)
│   ├── raw/images/                   # downloaded photos (gitignored)
│   ├── selected_species.csv          # chosen species (from notebook 01)
│   ├── splits.csv                    # train/val/test assignment (from notebook 02)
│   └── inat_taxon_map.json           # name→iNat taxon cache (gitignored)
├── notebooks/
│   ├── 01_eda_species.ipynb          # species selection            (maintainer only)
│   ├── 02_eda_images.ipynb           # image EDA + split            (maintainer only)
│   ├── 03_train_classifier.ipynb     # ResNet-50 baseline (fixed reference)
│   ├── 04_train_improved.ipynb       # ResNet-50 with fine-grained upgrades
│   ├── 04b_train_vit.ipynb           # ViT-base challenger (HF transformers)
│   ├── 05_evaluate.ipynb             # metrics + visuals on a saved checkpoint
│   ├── 06_detect_subject.ipynb       # zero-shot YOLO subject localizer + Δtop-1 (optional)
│   ├── 07_train_segmentation.ipynb   # distilled plant/bg segmenter (DeepLabV3-MNv3)
│   ├── 08_style_baseline.ipynb       # stylize baseline scaffold (placeholder)
│   ├── 09_style_retrieval.ipynb      # stylize classify→retrieve→generate scaffold (placeholder)
│   ├── 10_garden_layout.ipynb        # compose: rule plans + AR layout transformer
│   ├── 11_complete_layout.ipynb      # compose: masked-diffusion completion (pin + infill)
│   ├── 12_render_plan.ipynb          # compose: seg-controlnet render of any plan
│   ├── 13_refine_layout.ipynb        # compose: designed corpus + count tokens + best-of-N
│   ├── 14_real_plans.ipynb           # compose: hough-labeled real drawings → transfer
│   ├── 15_layered_layout.ipynb       # compose: layered corpus + literature metrics + calibration
│   └── 16_curated_design.ipynb       # compose: design-layer reframe + realism metric + blinded pairs
├── bvtrain/                          # shared training plumbing the notebooks import (env · data · checkpoint · fit · fit_seg · garden)
├── share/                            # the team model-sharing loop (publish/leaderboard/score)
├── scripts/
│   ├── download_inaturalist.py       # resumable, threaded downloader
│   ├── upload_to_hf.py               # publish dataset to Hugging Face
│   ├── generate_pseudo_masks.py      # FastSAM → masks companion HF dataset (maintainer)
│   ├── measure_delta_top1.py         # Δtop-1 with/without seg preprocessing
│   ├── precompute_student_masks.py   # student masks for the BG_AUG training augmentation (local path)
│   ├── publish_student_masks_full.py # paired {image, mask} HF companion dataset (HF path)
│   ├── harvest_real_plans.py         # crawl publisher plan drawings → data/real_plans/
│   ├── harvest_permapeople.py        # structured community plans → data/permapeople/
│   ├── permapeople_plans.py          # parse + score real plans (metric calibration)
│   └── measure_species_colors.py     # measured bloom/leaf colors → bvtrain/species_colors.json
├── checkpoints/                      # training scratch (gitignored)
├── models/planned.json              # roadmap rows for models not yet built
├── demo/                            # separate showcase app (React + minimal FastAPI)
├── kaggle/                          # headless Kaggle-GPU run tooling (kernel-metadata + README)
├── Makefile · requirements.txt
```

## AI assistance

Built with help from Claude Code (Claude Opus 4.8, Anthropic) — most heavily the
`demo/` app (React frontend + serving backend), plus repo plumbing and docs.
Model choices, experiments, evaluations, and conclusions were directed and
verified by the team.
