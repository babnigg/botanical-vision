# Run training on Kaggle GPU (headless, from VS Code)

Kaggle gives a free **P100** (~30 h/week, a separate quota from Colab) but has no VS Code
integration — so we push the notebook with the **Kaggle API** and pull results back, all
from the VS Code terminal. The notebook is self-contained: it clones `bvtrain` and loads
the dataset from Hugging Face, so nothing else needs uploading.

## One-time setup
1. `pip install kaggle`
2. Kaggle → Account → **Create New API Token** → save `kaggle.json` to
   `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`.
3. In `kaggle/kernel-metadata.json`, set `"id"` to `"<your-kaggle-username>/botanical-vision-train"`.

## Each run (from `code/project`, in the VS Code terminal)
```bash
cp notebooks/04_train_improved.ipynb kaggle/                       # ship the latest notebook
kaggle kernels push -p kaggle                                      # start it on Kaggle's P100
kaggle kernels status <username>/botanical-vision-train            # queued / running / complete
kaggle kernels output <username>/botanical-vision-train -p kaggle/out   # pull checkpoints + logs
```
The `*_best.pt` / `*_last.pt` checkpoints land in `kaggle/out/`. Bring one home and share it
with `python -m share.publish --checkpoint kaggle/out/resnet50_improved_best.pt --name my-model`.

## Resuming across sessions
A Kaggle GPU session is ~9–12 h. If a run doesn't finish, continue it:
1. After the first run completes, mount its own output as an input so its checkpoint is visible —
   set `"kernel_sources": ["<username>/botanical-vision-train"]` in `kernel-metadata.json`.
2. `cp` + `push` again. The notebook auto-finds `/kaggle/input/**/resnet50_improved_last.pt` and resumes.

## Notes
- GPU + Internet are enabled via `kernel-metadata.json` — don't remove those.
- To train the baseline instead, set `"code_file"` to `03_train_classifier.ipynb`.
- The notebook's bootstrap clones a **public** GitHub repo; a private repo would need a token.
