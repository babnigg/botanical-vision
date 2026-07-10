# Botanical Vision

Fine-grained flowering-plant classification and botanical-illustration generation
for Advanced Computer Vision (ADSP 32023).

The current stage builds the classification dataset: photos of flowering plants
labeled by species, sourced from citizen-science observations on iNaturalist.

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
ds = load_dataset("<username>/botanical-vision")   # train / val / test
```

The dataset is public, so no Hugging Face account or login is required. The first
call caches images locally (~25 GB when complete); for a quick look without the full
download, pass `streaming=True`.

**Torch / GPU.** `requirements.txt` installs a CPU build of PyTorch, which is enough
to run everything (slowly). For GPU training, install the CUDA build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Note: PyTorch must be **≥ 2.3** to work with NumPy 2. The notebooks pick GPU
automatically when available and fall back to CPU otherwise; Google Colab is a free
GPU option.

### Train a classifier

Two notebooks, both fine-tuning ResNet-50 on the species splits (`data/splits.csv`)
and reporting top-1 / top-5 accuracy:

- `notebooks/03_train_classifier.ipynb` — the **baseline**: a plain fine-tune, kept
  fixed as the comparison point.
- `notebooks/04_train_improved.ipynb` — mirrors the baseline with fine-grained
  upgrades: stronger augmentation (RandAugment, color jitter, random erasing), label
  smoothing, discriminative learning rates, a cosine schedule, more epochs, and
  best-val checkpointing.

Both have an `N_SPECIES` toggle — set it to an int (e.g. 100) for a quick run, or
leave it `None` to train on all species (a multi-hour run on the full dataset).

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

```bash
huggingface-cli login
python scripts/upload_to_hf.py --repo <username>/botanical-vision
```

Reads `splits.csv` and `selected_species.csv`, builds the train/val/test
`DatasetDict` (each image labeled by species, with genus/family/order/class
metadata), pushes it to the Hub, and writes the dataset card. Defaults to public;
pass `--private` for a private dataset (which then requires teammates to have an
account and be granted access). To smoke-test the upload on a handful of species,
pass `--splits-csv <small_split.csv>` pointing at a mini split.

The full dataset is ~25 GB, so the push takes a while. It's **resumable and retries
through brief internet outages** — already-uploaded shards are skipped, so it's safe
to re-run (or let it wait out) a connection that drops for a minute at a time.

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
│   ├── 01_eda_species.ipynb          # species selection
│   ├── 02_eda_images.ipynb           # image EDA + split
│   ├── 03_train_classifier.ipynb     # ResNet-50 baseline (fixed reference)
│   ├── 04_train_improved.ipynb       # ResNet-50 with fine-grained upgrades
│   └── 05_evaluate.ipynb             # metrics + visuals on a saved checkpoint
├── scripts/
│   ├── download_inaturalist.py       # resumable, threaded downloader
│   └── upload_to_hf.py               # publish dataset to Hugging Face
├── checkpoints/                      # saved models (gitignored)
└── requirements.txt
```
