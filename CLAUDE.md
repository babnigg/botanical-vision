# Repo guide for AI assistants (and humans in a hurry)

**Project:** ADSP 32023 (Advanced Computer Vision). Fine-grained plant classification +
botanical-illustration generation + sketch-to-landscape. The **CV work is the graded
core**; `api/`, `app/`, and `mlops/` are supporting best-practice infrastructure — keep
them working but don't let infra churn overshadow CV substance.

Read `CONTRIBUTING.md` for the full workflow. The hard rules:

- **The team loop is just two commands**: `mlops.publish` (share a trained model to the
  shared HF model repo) and `mlops.leaderboard` (compare everyone's on the same test
  split). Keep this simple — don't make teammates run MLflow/Prefect/Docker to share or
  compare models. Those are behind-the-scenes and only concern the demo host.
- **Keep `README.md` and `.gitignore` up to date with every change** (shared repo).
- Run `make lint` and `make test` before proposing a commit; CI must pass.
- Models reach the app only via the registry: **train → `mlops.register_checkpoint` →
  `mlops.evaluate` → `mlops.promote` (champion/challenger gate) → served as
  `@production`**. Status is read from MLflow, never hand-edited.
- Adding another section's model = register it + add a row to `mlops/config.py` `MODELS`
  (+ its own pyfunc flavor and a model-appropriate eval). Don't reuse the classifier's
  top-1 eval for generators.
- Never commit weights (`*.pt`), data/images, `mlflow.db`, `mlruns/`, `.env`,
  `node_modules/`. Data lives on Hugging Face; a shared registry is Databricks (one env var).
- Serve from `models:/botanical-classifier@production`; the API falls back to a labeled
  stub when no model/torch is available, so the frontend always runs.
