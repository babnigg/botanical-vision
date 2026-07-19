"""bvtrain — shared training/eval plumbing for the Botanical Vision notebooks.

The notebooks (03 / 04 / 05) stay focused on what you actually customize: the model,
the augmentation, the optimizer. Everything else lives here — environment detection,
data loading, VRAM-aware batching + gradient accumulation, resumable checkpointing
across local / Colab / Kaggle, the training loop, and top-1/5 evaluation.

    import bvtrain as bv
    env     = bv.setup()
    data    = bv.load_data(env)
    loaders = bv.build_loaders(data, train_tf, env)
    hist    = bv.fit(model, optimizer, loaders, epochs=15, run_name="resnet50_improved",
                     env=env, scheduler=scheduler, criterion=criterion)
    bv.plot_history(hist)
    bv.evaluate(model, loaders.test, env, run_name="resnet50_improved")
"""
from .data import (
    IMG_SIZE,
    MEAN,
    STD,
    Data,
    Loaders,
    build_loaders,
    eval_transforms,
    load_data,
)
from .env import Env, setup
from .train import checkpoint_dir, evaluate, fit, load_model, plot_history

__all__ = [
    "Env", "setup", "MEAN", "STD", "IMG_SIZE", "Data", "Loaders",
    "eval_transforms", "load_data", "build_loaders",
    "fit", "evaluate", "plot_history", "load_model", "checkpoint_dir",
]
