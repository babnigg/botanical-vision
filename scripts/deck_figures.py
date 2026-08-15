"""Presentation figures in the deck theme (cream/green/gold, EB Garamond).

Reproducible: everything derives from repo data or recorded results. Transparent
backgrounds so figures embed on the slide paper. Rebuild with:

    python scripts/deck_figures.py [--out <dir>]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_DIR = Path(__file__).resolve().parent.parent
INK, GREEN, SAGE, GOLD = "#2F3527", "#3F5237", "#76835B", "#A8863E"
SAGE_SOFT = "#76835B55"

for name in ("EBGaramond.ttf", "EBGaramond-Italic.ttf", "PlayfairDisplay.ttf"):
    p = os.path.expandvars(rf"%LOCALAPPDATA%\Microsoft\Windows\Fonts\{name}")
    if os.path.exists(p):
        fm.fontManager.addfont(p)
if any("EB Garamond" in f.name for f in fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "EB Garamond"
plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": "none",
                     "savefig.transparent": True, "text.color": INK,
                     "axes.edgecolor": SAGE, "xtick.color": INK, "ytick.color": INK,
                     "font.size": 15})


def _despine(ax, keep=("bottom",)):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def fig_iteration_arc(out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.6), width_ratios=[5, 3.2],
                                 gridspec_kw={"wspace": 0.16})
    labels = ["random", "AR\ntransformer", "masked\ndiffusion", "+ search\n(best-of-8)", "rule\nteacher"]
    vals = [0.63, 0.76, 0.78, 0.89, 0.90]
    cols = [SAGE_SOFT, SAGE_SOFT, SAGE_SOFT, GOLD, GREEN]
    bars = a1.bar(labels, vals, color=cols, width=0.6, edgecolor=INK, linewidth=0.7)
    for r, v in zip(bars, vals):
        a1.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center",
                fontsize=16, color=INK)
    a1.set_ylim(0, 1.04)
    a1.set_yticks([])
    a1.set_xlabel("rule-metric score, higher is better", fontsize=13.5, labelpad=10)
    labels2 = ["v3\ncorpus", "random", "model\n(best-of-24)", "curated\nteacher"]
    vals2 = [3.63, 3.51, 1.16, 0.81]
    cols2 = [SAGE_SOFT, SAGE_SOFT, GOLD, GREEN]
    bars2 = a2.bar(labels2, vals2, color=cols2, width=0.6, edgecolor=INK, linewidth=0.7)
    for r, v in zip(bars2, vals2):
        a2.text(r.get_x() + r.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center",
                fontsize=16, color=INK)
    a2.set_ylim(0, 4.15)
    a2.set_yticks([])
    a2.set_xlabel("distance to real beds, lower is better", fontsize=13.5, labelpad=10)
    for a in (a1, a2):
        _despine(a)
        a.tick_params(length=0, labelsize=14)
    plt.tight_layout()
    plt.savefig(out / "iteration_arc.png", dpi=180)
    plt.close()


def fig_project_map(out: Path):
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#3F523710", ec=GREEN, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 0.5, title, ha="center", fontsize=17, color=tc,
                family="Playfair Display" if any("Playfair" in f.name for f in fm.fontManager.ttflist) else None)
        ax.text(x + w / 2, y + (h - 0.75) / 2, sub, ha="center", va="center",
                fontsize=12.5, color=tc, linespacing=1.5)

    def arrow(x1, y1, x2, y2, label=None, lx=0.0, ly=0.14):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=17, lw=1.7, color=SAGE,
                                     shrinkA=5, shrinkB=5, alpha=0.85))
        if label:
            ax.text((x1 + x2) / 2 + lx, max(y1, y2) + ly, label, fontsize=11.5,
                    color=SAGE, style="italic", ha="center")

    box(4.1, 4.3, 4.4, 1.3, "the dataset",
        "4,094 species · 408K iNaturalist photos\nGBIF selection → dedup → HF hub")
    box(0.6, 1.85, 5.1, 1.85, "problem 1 · identify",
        "fine-grained classifier (ResNet-50, ViT)\ndetection & segmentation experiments\nbaseline → champion, every step measured")
    box(6.9, 1.85, 5.1, 1.85, "problem 2 · compose",
        "planting plans as (species, x, y, spread)\nAR → masked diffusion → curated design\njudged against real garden beds")
    box(3.6, 0.2, 5.4, 1.05, "the demo",
        "identify → toolbox → pin → compose → render", fc=GREEN, ec=GREEN, tc="#F5EFE2")
    arrow(5.3, 4.3, 3.5, 3.7, "trains", lx=-0.75, ly=-0.1)
    arrow(7.3, 4.3, 9.1, 3.7, "photos seed the renders", lx=1.15, ly=-0.1)
    arrow(3.2, 1.85, 5.1, 1.25)
    arrow(9.4, 1.85, 7.5, 1.25)
    plt.tight_layout()
    plt.savefig(out / "project_map.png", dpi=180)
    plt.close()


def fig_class_longtail(out: Path):
    # the real long tail: GBIF observation counts, which motivated the >=2,000 cutoff
    gbif = pd.read_csv(PROJECT_DIR / "data" / "raw" / "gbif_species_list.csv", sep="\t",
                       usecols=["numberOfOccurrences", "taxonRank", "taxonomicStatus", "speciesKey"],
                       on_bad_lines="skip")
    gbif = gbif[(gbif.taxonRank == "SPECIES") & (gbif.taxonomicStatus == "ACCEPTED")]
    gbif = gbif.drop_duplicates("speciesKey")           # mirrors notebook 01's selection
    counts = np.sort(gbif["numberOfOccurrences"].dropna().astype(int).values)[::-1]
    ranks = np.arange(1, len(counts) + 1)
    n_sel = int((counts >= 2000).sum())
    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    ax.fill_between(ranks[:n_sel], counts[:n_sel], 1, color=GOLD, alpha=0.30, lw=0)
    ax.fill_between(ranks[n_sel:], counts[n_sel:], 1, color=SAGE, alpha=0.25, lw=0)
    ax.plot(ranks, counts, color=GREEN, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlim(1, len(counts))
    ax.set_ylim(1, counts.max() * 1.5)
    ax.axhline(2000, color=GOLD, lw=1.7, ls=(0, (5, 4)))
    ax.text(len(counts) * 0.98, 2600, "2,000-observation cutoff", ha="right",
            fontsize=14, color=GOLD)
    ax.text(n_sel * 0.45, counts[int(n_sel * 0.45)] * 2.6,
            f"{n_sel:,} species selected", fontsize=14.5, color=INK)
    ax.set_xlabel("flowering-plant species, ranked by iNaturalist observations", fontsize=14.5)
    ax.set_ylabel("observations (log)", fontsize=14.5)
    _despine(ax, keep=("bottom", "left"))
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.savefig(out / "eda_longtail.png", dpi=180)
    plt.close()


def fig_ops_map(out: Path):
    fig, ax = plt.subplots(figsize=(12.6, 5.2))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def box(x, y, w, h, title, sub, fc="#3F523710", ec=GREEN, tc=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                                    fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 0.45, title, ha="center", fontsize=15, color=tc,
                family="Playfair Display" if any("Playfair" in f.name for f in fm.fontManager.ttflist) else None)
        ax.text(x + w / 2, y + (h - 0.65) / 2, sub, ha="center", va="center",
                fontsize=11.5, color=tc, linespacing=1.45)

    def arrow(x1, y1, x2, y2, label=None, ly=0.16):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=1.7, color=SAGE,
                                     shrinkA=5, shrinkB=5, alpha=0.85))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + ly, label, fontsize=11,
                    color=SAGE, style="italic", ha="center")

    box(0.5, 3.4, 3.6, 1.5, "train anywhere",
        "notebooks · kaggle headless\nresumable checkpoints")
    box(4.7, 3.4, 3.4, 1.5, "publish",
        "python -m share.publish\nlean bundle → HF model repo")
    box(8.8, 3.4, 3.3, 1.5, "leaderboard",
        "same streamed test protocol\nfor every member's model")
    box(4.7, 0.4, 3.4, 1.4, "the demo",
        "serves the champion\ncheckpoint directly", fc=GREEN, ec=GREEN, tc="#F5EFE2")
    box(0.5, 0.4, 3.6, 1.4, "data by script",
        "dataset · masks · plan corpus\nall regenerate from scripts")
    arrow(4.1, 4.15, 4.7, 4.15)
    arrow(8.1, 4.15, 8.8, 4.15)
    arrow(10.4, 3.4, 6.6, 1.8, "champion = promotion")
    plt.tight_layout()
    plt.savefig(out / "ops_map.png", dpi=180)
    plt.close()


def fig_taxonomy_ladder(out: Path):
    # where predictions land (fp32 full-vocab eval, notebook 05):
    # exact 0.625, right genus 0.684, right family 0.739
    segs = [("exact species", 0.625, GREEN),
            ("right genus", 0.059, SAGE),
            ("right family", 0.055, GOLD),
            ("wrong family", 0.261, "#B7AB90")]
    fig, ax = plt.subplots(figsize=(11.5, 2.1))
    x = 0.0
    for name, wdt, col in segs:
        ax.barh(0, wdt, left=x, height=0.52, color=col, edgecolor="none")
        inside = wdt > 0.12
        tcol = "#F5EFE2" if name == "exact species" else INK
        if inside:
            ax.text(x + wdt / 2, 0, name + "\n" + format(wdt, ".0%"), ha="center",
                    va="center", fontsize=15, color=tcol, linespacing=1.25)
        elif name == "right genus":                     # stagger the two thin segments
            ax.text(x + wdt / 2, 0.5, "right genus +" + format(wdt, ".0%"),
                    ha="right", va="bottom", fontsize=12.5, color=INK)
        else:
            ax.text(x + wdt / 2, -0.5, "right family +" + format(wdt, ".0%"),
                    ha="left", va="top", fontsize=12.5, color=INK)
        x += wdt
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.9, 0.85)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out / "taxonomy_ladder.png", dpi=180)
    plt.close()


def fig_random_vs_rules(out: Path):
    import random as _r
    import sys
    sys.path.insert(0, str(PROJECT_DIR))
    from bvtrain import garden as g
    _r.seed(5)
    w, d, sun = 6.0, 2.4, 2
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.8, 5.8))
    g.show_plan(g.gen_random(w, d), w, d, a1, "random placement")
    g.show_plan(g.gen_plan(w, d, sun), w, d, a2, "rule generator")
    for a in (a1, a2):
        a.set_facecolor("none")
        a.title.set_fontsize(15)
        for sp in a.spines.values():
            sp.set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "layout_random_vs_rules.png", dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "Downloads" / "deck-figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_iteration_arc(out)
    fig_project_map(out)
    fig_ops_map(out)
    fig_taxonomy_ladder(out)
    fig_random_vs_rules(out)
    try:
        fig_class_longtail(out)
    except FileNotFoundError:
        print("splits.csv not found — eda_longtail skipped")
    print("deck figures ->", out)


if __name__ == "__main__":
    main()
