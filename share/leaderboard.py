"""Compare everyone's shared models — one command.

Pulls every model teammates published to the shared Hugging Face repo, scores each on
the *same* held-out test split, and prints a ranked table. This is the payoff of the
whole setup: an apples-to-apples comparison so the team can see whose model is best.

    python -m share.leaderboard --limit 300
"""
from __future__ import annotations

import argparse

from . import hub
from .model import build_predictor
from .score import score


def leaderboard(limit: int = 300) -> list[dict]:
    models = hub.list_models()
    if not models:
        print(f"no shared models yet in {hub.SHARED_REPO}.")
        print("be the first: python -m share.publish --checkpoint <ckpt> --name <name>")
        return []

    rows = []
    for m in models:
        path = hub.pull(m["author"], m["name"])
        res = score(build_predictor(path), limit)
        rows.append({**m, **res})
        print(f"  scored {m['author']}/{m['name']}: top1={res['top1']:.3f}")
    rows.sort(key=lambda r: r["top1"], reverse=True)

    print(f"\n{'#':<4}{'model':<30}{'top-1':>8}{'top-5':>8}   notes")
    for i, r in enumerate(rows, 1):
        tag = "   <- champion" if i == 1 else ""
        name = f"{r['author']}/{r['name']}"
        print(f"{i:<4}{name:<30}{r['top1']:>8.3f}{r['top5']:>8.3f}   {r.get('notes', '')}{tag}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank the team's shared models on the same test split.")
    ap.add_argument("--limit", type=int, default=300, help="test images per model (higher = slower, more exact)")
    args = ap.parse_args()
    leaderboard(args.limit)


if __name__ == "__main__":
    main()
