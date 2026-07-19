"""The team model-sharing loop — the whole point of the repo for the class.

Everyone trains better models in the notebooks, then shares their best here so the
team can compare them on the same test split. Two commands:

    python -m share.publish --checkpoint checkpoints/<ckpt>.pt --name my-model
    python -m share.leaderboard

No MLflow, no registry, no server — models travel via a shared Hugging Face model
repo (weights never go in git). First time only: ``huggingface-cli login``.
"""
