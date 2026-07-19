# Botanical Vision

Fine-grained flowering-plant classification for Advanced Computer Vision (ADSP 32023) —
a gardener's tool with a computer-vision spine: **Identify** a plant from a photo,
collect species in a **Toolbox**, and **Compose** a garden design.

The current stage builds and improves the classifier: photos of flowering plants labeled
by species, sourced from citizen-science observations on iNaturalist.

## What we're actually doing (start here)

This is a **computer-vision class**, so the only work that matters is **making the model
better**. The whole loop is three steps, and only the middle one is really yours:

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

Teammates run only **03 → 05** — they auto-load the dataset from Hugging Face, so you
never need local data.

| Notebook | Who | What to do |
| --- | --- | --- |
| `01_eda_species` | maintainer only | **don't re-run** — rebuilds the species selection; read only |
| `02_eda_images` | maintainer only | **don't re-run** — inventories raw images + writes the split; read only |
| `03_train_classifier` | everyone | **run-only** baseline (fixed reference) — don't experiment here |
| `04_train_improved` | everyone | **where model work goes** — all improvements live here |
| `05_evaluate` | everyone | metrics/visuals on a trained checkpoint |

Training writes a `.pt` to `checkpoints/`. To share it with the team, run
`python -m share.publish` (see *start here* above).

### Train a classifier

Two notebooks, both fine-tuning ResNet-50 on the species splits (`data/splits.csv`)
and reporting top-1 / top-5 accuracy:

- `notebooks/03_train_classifier.ipynb` — the **baseline**: a plain fine-tune, kept
  fixed as the comparison point (run it, don't modify it).
- `notebooks/04_train_improved.ipynb` — the **active model** where improvements go:
  stronger augmentation (RandAugment, color jitter, random erasing), label
  smoothing, discriminative learning rates, a cosine schedule, more epochs, and
  best-val checkpointing.

Both have an `N_SPECIES` toggle — set it to an int (e.g. 100) for a quick run, or
leave it `None` to train on all species (a multi-hour run on the full dataset).

**Resumable training.** Both save full state (model, optimizer, scheduler, epoch,
step, best-val, history) every `CKPT_EVERY` steps and write two checkpoints per run:
`{RUN_NAME}_last.pt` (rolling) and `{RUN_NAME}_best.pt` (best val). On restart they
auto-resume from `_last.pt` if its config signature matches the current setup — even
mid-epoch — so a long run survives interruptions. A changed config (species count,
epochs) starts fresh; set `FRESH = True` to force a fresh run. On **Google Colab**,
local disk is wiped on disconnect, so mount Drive and point `CKPT_DIR` at a Drive
path (the checkpoint cell shows how) or your progress won't survive.

`05_evaluate.ipynb` loads `{RUN_NAME}_best.pt` — set `CHECKPOINT` there to the model
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
│   └── 05_evaluate.ipynb             # metrics + visuals on a saved checkpoint
├── bvtrain/                          # shared training plumbing the notebooks import (env · data · checkpoint · fit)
├── share/                            # the team model-sharing loop (publish/leaderboard/score)
├── scripts/
│   ├── download_inaturalist.py       # resumable, threaded downloader
│   └── upload_to_hf.py               # publish dataset to Hugging Face
├── checkpoints/                      # training scratch (gitignored)
├── models/planned.json              # roadmap rows for models not yet built
├── demo/                            # separate showcase app (React + minimal FastAPI)
├── kaggle/                          # headless Kaggle-GPU run tooling (kernel-metadata + README)
├── Makefile · requirements.txt
```
