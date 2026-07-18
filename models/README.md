# Models — sharing, comparing, and serving

## Sharing & comparing (what everyone uses)

The team shares trained models through a **shared Hugging Face model repo**
(`dbabnigg/botanical-vision-models`) — the model equivalent of how the dataset already
lives on the Hub. Weights never go in git.

```bash
python -m mlops.publish --checkpoint checkpoints/resnet50_improved_best.pt --name my-model
python -m mlops.leaderboard        # pulls everyone's, scores on the same test split, ranks
```

`publish` uploads a lean bundle (weights + labels + transform) under `{your-username}/{name}`;
`leaderboard` downloads every shared model and evaluates them identically. That's the whole
model-improvement loop for the class — the MLflow machinery below is **not** part of it.

---

## Serving a champion (behind the scenes — demo host only)

The app serves models from a real **MLflow Model Registry**, not a hand-edited file.
Training tracks runs in MLflow; a checkpoint is packaged as a versioned pyfunc model;
an evaluation gate promotes a version to `@production`; the API serves whatever holds
that alias. This mirrors the ADSP 32021 pipeline (train → track/register → gate →
serve), using MLflow 3 **aliases** in place of the deprecated Staging/Production stages.

## Where things live (Core / laptop tier)
- Tracking + registry backend: `sqlite:///mlops/mlflow.db` (gitignored)
- Model artifacts: `mlruns/` (gitignored)
- Config (env-driven): `mlops/config.py` — set `MLFLOW_TRACKING_URI=databricks` for a shared registry.
- `models/planned.json` — roadmap rows for models **not yet trained** (generator/data). Trained models come from MLflow, not this file.

## The model is self-describing
`mlops/model.py` packages ResNet-50 as an `mlflow.pyfunc` "flavor" that bundles the eval
transform (Resize 256 / CenterCrop 224 / Normalize) **and** the 4,094-label vocabulary
with the weights. The serving side never re-implements preprocessing — the course's
anti-train/serve-skew pattern. Input: base64 image(s); output: top-5 `{species, confidence}`.

## Lifecycle
1. **Train** — notebooks 03/04 (tracked in MLflow).
2. **Register** — `python -m mlops.register_checkpoint --checkpoint <ckpt> [--alias staging]`
   → a new version of `botanical-classifier`.
3. **Evaluate** — `python -m mlops.evaluate --version <v> --limit 300` (streamed HF test split → top-1/top-5).
4. **Gate / promote** — `python -m mlops.promote --challenger <v> --margin 0.005`
   → sets `@production` **only if** it beats the current champion (champion/challenger).
5. **Serve** — the API loads `models:/botanical-classifier@production` automatically.
6. **Orchestrate** — `python -m mlops.flows.train_flow --checkpoint <ckpt>` runs register → evaluate → gate as a Prefect flow.

## Status mapping (roadmap UI)
| MLflow state | roadmap |
| --- | --- |
| version holds `@production` | **live** |
| registered, not promoted | **candidate** |
| in `planned.json`, not in MLflow | **planned** |

## Aliases vs stages
MLflow 3 deprecated the `Staging`/`Production` *stages* the 32021 lectures show, in favor
of version *aliases* — same concept, current API. We use `production` / `staging`;
`models:/botanical-classifier@production` is the served model.

## Next steps (out of Core scope)
A shared team registry via **Databricks Managed MLflow** (one env var — matches the 32021
labs) or a self-hosted tracking server + MinIO/S3 artifacts so teammates share versions
and the Docker container can resolve artifacts; Evidently drift monitoring; canary/
blue-green rollout.
