"""bvtrain — shared training/eval plumbing for the Botanical Vision notebooks.

The notebooks (03-12) stay focused on what you actually customize: the model, the
augmentation, the optimizer. Everything else lives here — environment detection, data
loading, VRAM-aware batching + gradient accumulation, resumable checkpointing across
local / Colab / Kaggle, the training loop, and evaluation (top-1/5 for classification;
mean IoU for segmentation). Compose domain primitives (plant palette, rule generator,
layout metrics, plan drawing) live in `bvtrain.garden` — notebooks 10-12 import it as
`from bvtrain import garden as g`.

Classification (notebooks 03/04, eval 05):

    import bvtrain as bv
    env     = bv.setup()
    data    = bv.load_data(env)
    loaders = bv.build_loaders(data, train_tf, env)
    hist    = bv.fit(model, optimizer, loaders, epochs=15, run_name="resnet50_improved",
                     env=env, scheduler=scheduler, criterion=criterion)
    bv.plot_history(hist)
    bv.evaluate(model, loaders.test, env, run_name="resnet50_improved")

Segmentation (notebook 07):

    seg     = bv.load_seg_data(env, masks_repo="<user>/botanical-vision-256-masks")
    loaders = bv.build_seg_loaders(seg, train_tf, env)
    hist    = bv.fit_seg(model, optimizer, loaders, epochs=10, run_name="deeplabv3_mnv3_seg",
                         env=env, scheduler=scheduler)
    bv.evaluate_seg(model, loaders.test, env, run_name="deeplabv3_mnv3_seg")
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
from .seg_data import (
    SegData,
    SegLoaders,
    build_seg_loaders,
    eval_seg_transforms,
    load_seg_data,
)
from .seg_train import (
    BceDiceLoss,
    dice_loss,
    evaluate_seg,
    fit_seg,
    load_seg_model,
    plot_seg_history,
)
from .train import checkpoint_dir, evaluate, fit, load_model, plot_history

__all__ = [
    "Env", "setup", "MEAN", "STD", "IMG_SIZE", "Data", "Loaders",
    "eval_transforms", "load_data", "build_loaders",
    "fit", "evaluate", "plot_history", "load_model", "checkpoint_dir",
    "SegData", "SegLoaders", "eval_seg_transforms", "load_seg_data",
    "build_seg_loaders", "BceDiceLoss", "dice_loss", "fit_seg",
    "evaluate_seg", "load_seg_model", "plot_seg_history",
]
