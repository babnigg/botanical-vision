"""Champion / challenger promotion gate.

The Session-4 "evaluation as a pipeline gatekeeper": evaluate the challenger on the
same holdout as the current production champion, and move the ``production`` alias to
the challenger **only if** it beats the champion's top-1 by ``--margin`` (or meets
``--min-top1`` when there is no champion). Exposes ``promote()`` for the flow; the CLI
exits non-zero on a failed gate so CI/orchestration fails loudly.

    python -m mlops.promote --challenger 2 --margin 0.005 --limit 300
"""
from __future__ import annotations

import argparse

from mlops import config
from mlops.evaluate import evaluate


def promote(challenger: str, margin: float = 0.0, min_top1: float = 0.0, limit: int = 300) -> dict:
    config.configure()
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    ch = evaluate(f"models:/{config.MODEL_NAME}/{challenger}", limit)

    champ_v = None
    try:
        champ_v = client.get_model_version_by_alias(config.MODEL_NAME, "production").version
    except Exception:
        pass

    champ = None
    if champ_v and str(champ_v) != str(challenger):
        champ = evaluate(f"models:/{config.MODEL_NAME}/{champ_v}", limit)
        ok = ch["top1"] >= champ["top1"] + margin
        reason = f"challenger {ch['top1']:.3f} vs champion {champ['top1']:.3f} + margin {margin}"
    else:
        ok = ch["top1"] >= min_top1
        reason = f"challenger {ch['top1']:.3f} vs min {min_top1} (no champion)"

    if ok:
        client.set_registered_model_alias(config.MODEL_NAME, "production", challenger)
    return {"promoted": ok, "challenger": ch, "champion": champ, "champ_version": champ_v, "reason": reason}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenger", required=True)
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--min-top1", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()

    r = promote(args.challenger, args.margin, args.min_top1, args.limit)
    ch = r["challenger"]
    print(f"challenger v{args.challenger}: top1={ch['top1']:.3f} top5={ch['top5']:.3f} (n={ch['n']})")
    if r["champion"]:
        print(f"champion  v{r['champ_version']}: top1={r['champion']['top1']:.3f}")
    if r["promoted"]:
        print(f"GATE PASS - promoted v{args.challenger} to @production ({r['reason']})")
    else:
        print(f"GATE FAIL - v{args.challenger} left unpromoted ({r['reason']})")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
