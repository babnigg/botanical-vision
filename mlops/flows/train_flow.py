"""Prefect flow: train -> register -> evaluation gate.

The 'pipeline is the product' deliverable. Register/evaluate/gate are fully
automated; the train step passes through an existing checkpoint in the Core tier
(swap in the extracted training loop to make it retrain). Run locally:

    pip install prefect
    python -m mlops.flows.train_flow --checkpoint checkpoints/resnet50_improved.pt --margin 0.005 --limit 300
"""
from __future__ import annotations

import argparse

from prefect import flow, get_run_logger, task

from mlops.promote import promote as _promote
from mlops.register_checkpoint import register as _register
from mlops.train import train as _train


@task
def train_task(checkpoint: str | None, n_species: int | None, epochs: int | None) -> str:
    return _train(checkpoint=checkpoint, n_species=n_species, epochs=epochs)


@task
def register_task(checkpoint: str) -> str:
    return _register(checkpoint)  # registers a new version, no alias yet


@task
def gate_task(version: str, margin: float, min_top1: float, limit: int) -> dict:
    return _promote(version, margin=margin, min_top1=min_top1, limit=limit)


@flow(name="botanical-train-and-promote")
def train_and_promote(checkpoint: str | None = None, n_species: int | None = None,
                      epochs: int | None = None, margin: float = 0.005,
                      min_top1: float = 0.0, limit: int = 300) -> dict:
    log = get_run_logger()
    ckpt = train_task(checkpoint, n_species, epochs)
    version = register_task(ckpt)
    log.info(f"registered version {version}; running the evaluation gate")
    result = gate_task(version, margin, min_top1, limit)
    log.info("PROMOTED" if result["promoted"] else "GATE FAILED — champion unchanged")
    return {"version": version, **{k: result[k] for k in ("promoted", "reason")}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-species", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--margin", type=float, default=0.005)
    ap.add_argument("--min-top1", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    out = train_and_promote(args.checkpoint, args.n_species, args.epochs,
                            args.margin, args.min_top1, args.limit)
    print(out)


if __name__ == "__main__":
    main()
