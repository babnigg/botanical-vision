# Botanical Vision — final presentation outline (15 min)

Fixed building block. Slides, claims, anchors and citations are stable; only the three
`[SLOT]` numbers get filled as the last runs land. Presentation-only deliverable (no
report); the outline absorbs the full rubric: EDA · two cognitive problems with
champion-challenger pairs · comparative metrics · model operations · conclusion.

**Frame (per instructor guidance): process and choices over model details.** Every
slide is a decision: what we faced → options → choice + why → measurement → lesson.

**Two pieces, three through-lines.** Act I = the classification pipeline. Act II = the
landscape generator. Connected by: (T1) *measure, don't assume* — every claim carries a
number; (T2) *distribution thinking* — what the model saw in training decides what
helps at inference (Week 4 transfer-learning framing); (T3) *data beats architecture* —
the corpus/teacher moved results more than any model swap.

Legend: **[C: …]** = course anchor (week/notebook/lecture). **[cite: …]** = outside
course scope, cite on slide. Metrics practice mirrors the course: accuracy + macro-F1 +
confusion matrix [C: 2.14-IRIS, 3.5-MNIST-SVM, HW1/HW2]; top-5 noted as standard
fine-grained practice beyond taught metrics.

---

## 0 · Title & abstract — 0:45
Problem, approach, headline results in three sentences. Two pillars named. Team.
*Asset: title over one Compose render.*

## 1 · Data & EDA: what the data forced us to decide — 1:30
- Pipeline: GBIF species list → iNaturalist photos → published on Hugging Face.
- Choices with reasons: ≥2,000-obs species cutoff (long tail), ≤100 imgs/species
  (balance), md5 dedup + cross-species leakage purge (5,302 leaked hashes), stratified
  70/15/15.
- EDA mirrors the taught pattern [C: Aircraft_EDA.ipynb, 3.1-ImageBasics, HW1]:
  class-distribution long-tail chart, resolution histogram, sample grid, dupe counts.
- 4,094 species · 408K images. *Assets: nb01 long-tail chart, nb02 sample grid + dupe table.*

## 2 · The map: two problems, three through-lines — 0:45
Cognitive problem 1: fine-grained classification. Cognitive problem 2: generative
layout synthesis. State T1–T3 in one line each; promise both acts will instantiate them.
*Asset: one diagram (pillars + demo roof).*

---

### ACT I — the classification pipeline

## 3 · Champion vs challenger: baseline → improved — 1:15
- Frozen-baseline discipline (choice: reproducibility over speed); improvements only in
  the challenger notebook. Champion-challenger form is the course's own
  [C: 3.8-CIFAR "Champion vs Challenger Bake Off"; transfer recipe C: 4.7/4.8, HW2].
- Numbers: baseline full-vocab top-1 0.549 / top-5 0.755 (Kaggle T4); improved
  100-species 0.704/0.897; **[SLOT: improved full-vocab from the running Kaggle job]**.
- Beyond-course ingredients flagged: label smoothing [cite: Szegedy et al. 2016],
  discriminative LRs [cite: Howard & Ruder 2018 (ULMFiT)]; cosine schedule is in-course
  [C: 6.2, 6.12].
- Eval per taught practice: accuracy, macro-F1, family-level confusion heatmap
  [C: classification_report/confusion pattern, 2.14/3.5/6.6]; t-SNE of embeddings
  [C: 7.5, 8.2]. *Assets: metrics table, nb05 confusion heatmap.*

## 4 · Choice story: "isolating the subject should help" — detection says no — 1:15
- The hypothesis everyone has. Options considered: fine-tune a detector (taught path
  [C: Aircraft_YOLO_Training]) — impossible, no box labels; so zero-shot open-vocab
  YOLO-World [cite: Cheng et al. 2024] with a Δtop-1 protocol instead of mAP (choice
  driven by what the data lacks — rubric's "pros/cons w.r.t. the data").
- Result: cropping hurts at every padding (best −4.0pp); ensemble ≈ 0; confidence
  gating fails. Lesson (T2): background context is signal the classifier trained on.
- Kept as a negative result on purpose (T1). *Asset: nb06 padding-sweep chart.*

## 5 · The echo: segmentation says no, louder — 1:15
- Design response to slide 4: soft-mask (blur background) instead of hard crop.
  Teacher–student distillation [C: mlops 4-Knowledge_Distillation; Wk4 p29] from SAM
  family [C: 5.10-Segmentation (SAM)] via FastSAM [cite: Zhao et al. 2023]; student
  DeepLabV3-MobileNetV3 [cite: Chen et al. 2017; Howard et al. 2019].
- Process beat: the first pseudo-label batch was ~50% garbage — caught by visual audit,
  fixed with a scored mask selector before any training (T1). Pseudo-labels themselves
  are beyond taught SSL [C: 8.8 self-distillation is the in-course neighbor;
  cite: Xie et al. 2020 (Noisy Student)].
- Measured: **−31.8pp top-1** (n=3,000) with the mechanism shown visually. What shipped
  anyway: the attention overlay in the demo. *Assets: teacher audit before/after, soft-mask
  mechanism image.*

## 6 · The redesign: move invariance into training — 1:00
- T2 applied: if masked inputs hurt at inference, make them part of training —
  background-blur augmentation (augmentation-as-strategy is in-course
  [C: Wk3 p48/53; Wk8 p13]); masks from our own student model.
- A/B at identical recipe/budget: **[SLOT: bg-aug vs control, from the running local A/B]**.
- Closes the act's arc: hypothesis → two measured failures → diagnosis → redesign →
  measurement. *Asset: A/B bar or table.*

## 7 · What the classifier unlocks: Stylize — 1:00
- Beyond style transfer (choice): classify → retrieve conspecifics from an embedding
  index [C: image-search ResNet50+Qdrant; HW3 precision@5/mAP@10] → generate new
  compositions of that species via SD img2img [C: 7.11] / IP-Adapter
  [cite: Ye et al. 2023].
- Judged by species-fidelity — the classifier grades the generator — plus CLIP
  image-text similarity [C: 8.3; CLIPScore proper cite: Hessel et al. 2021].
- **[SLOT: stylize showpieces + fidelity numbers]**. *Assets: field-of-the-flower grid.*

---

### ACT II — the landscape generator

## 8 · The symbolic choice — 1:15
- Problem 2: generate a planting plan. Published work is pixel-space GANs on ~100
  scraped images [cite: Senem et al., DLA 2024; Land 2025] — no plant identity.
- Our choice: plans as (species, x, y, spread) tuples — layout generation
  [cite: Gupta et al. 2021 (LayoutTransformer); Inoue et al. 2023 (LayoutDM)].
- No dataset exists → write horticultural rules that are simultaneously the training
  corpus generator and the metric suite (T3: the data IS the design surface).
  *Asset: rule-plan vs random side-by-side.*

## 9 · Champion vs challenger, driven by an interaction requirement — 1:30
- Champion: autoregressive layout transformer [attention mechanics C: 6.1, 6.12;
  AR image modeling cite: Chen et al. 2020 (ImageGPT)] — learns banding/spacing, but
  prefix-only conditioning can't pin plants, drift parity fails (0.43).
- Challenger: masked discrete diffusion [cite: Chang et al. 2022 (MaskGIT); Austin
  et al. 2021 (D3PM); guided-diffusion framing C: Wk7 pp54-56] — pinning by
  construction, sun conditioning 1.00.
- Then the cheapest big win: decode-time search — best-of-8 reranked by our metrics +
  constraint repair (the Wk7 evaluation-metrics discipline used as guidance [C: 7.13]):
  0.80 → 0.89 ≈ teacher 0.90. Count-token obedience measured honestly (0.72) → slot
  pinning kept (T1). *Asset: iteration arc chart 0.63→0.89.*

## 10 · Real drawings, renders, and an honest tie — 1:15
- Labeling real plans is week-3 vision: Sobel/Canny edge support [C: 3.2/3.3] around
  Hough circle proposals [cite: Duda & Hart 1972] + a four-signal genre gate, every
  threshold calibrated on labeled examples; validated P≈1.0/R 0.91–0.96 on rendered GT.
- Transfer pretraining vs scratch at equal budget: a measured tie (0.871/0.876 ≈) —
  corpus size gates transfer (T3); reported as-is (T1).
- Rendering: plans are segmentation maps → seg-ControlNet, zero training
  [C: 7.12 (canny+depth coded); seg conditioning cite: Zhang et al. 2023]; CLIP-scored
  styles [C: 8.3]. *Assets: gate sheet, render pair.*

---

## 11 · Model operations: how the team actually iterates — 1:00
- Taught pattern: MLflow registry with Production/Staging promotion [C: week 5 mlops
  module]. Our adaptation (choice + why): a leaner equivalent — weights on a shared HF
  model repo, `share.publish` → `share.leaderboard` scores everyone on the identical
  protocol, champion status = promotion (the 3.8 bake-off, made continuous).
- Maintenance/parameter-update plan: resumable checkpointing everywhere (mid-epoch,
  cross-session), retraining via the Kaggle headless flow, masks/corpora regenerate by
  script. Serving: demo app loads checkpoints directly (FastAPI — beyond the taught
  Streamlit pattern [C: dog-breed-classifier app.py], an engineering choice).
- Live demo beat: pin two hydrangeas → the model plants the bed. *Asset: ops diagram +
  demo screenshot.*

## 12 · Conclusion & references — 1:00
- Findings = the through-lines, now earned: context is signal (crop −4pp, mask −32pp,
  augmentation [SLOT]); measurement over intuition (five decisions reversed by numbers);
  data over architecture (designed corpus > model swap; corpus size gated transfer).
- Future: real-plan corpus growth, trait table, ViT backbone [C: 6.6], style LoRA
  [C: LoRA concept mlops-3; practice cite: Hu et al. 2021].
- References slide: all [cite] entries above + dataset (GBIF DOI 10.15468/dl.3hragg,
  iNaturalist) + reused patterns (ultralytics, diffusers, torchvision).

---

**Timing total:** 13:45 + transitions ≈ 15:00.
**Speaker split (3–4 members, ~4–5 min each):** A: slides 0–3 · B: 4–6 (the
detection/segmentation/augmentation arc) · C: 8–10 · D (or A): 7, 11–12.
**Stable [SLOT]s (content changes, structure doesn't):** improved full-vocab number
(slide 3) · bg-aug A/B (slide 6, echoed slide 12) · stylize showpieces (slide 7).
