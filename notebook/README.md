# Python code track

**Educational only. Not for clinical use.** Data licenses: [DATA_LICENSES.md](../DATA_LICENSES.md).

`code_track.ipynb` is an optional notebook for students who finish the web app early. It repeats the workshop experiments as Python code: class counts, features from a frozen MobileNetV3-Large, a logistic regression, the class-imbalance comparison, the "R" stamp shortcut, the adult-X-ray shift and its fix, a Grad-CAM heatmap, and five exercises at the end.

It reads the same images the web app uses (`../app/public/data`). The numbers come out close to the app's but not identical, because the notebook uses a different network (MobileNetV3-Large, 960 features) than the browser (MobileNet v2, 1,280 features).

## Open it

From the repository root, with the project's virtual environment (see "Rebuild the data" in the main README for creating it):

```bash
scripts/.venv/Scripts/pip install -r scripts/requirements.txt   # Windows; scripts/.venv/bin/pip elsewhere
cd notebook
../scripts/.venv/Scripts/jupyter lab code_track.ipynb            # or: jupyter notebook code_track.ipynb
```

Start Jupyter from the `notebook/` folder: the notebook finds the data through `../app/public/data`. Run the cells from the top with Shift+Enter, or choose Run, Run All Cells.

To re-run it from the command line and save the outputs in the file:

```bash
scripts/.venv/Scripts/python scripts/run_notebook.py
```

## How long it takes

Measured on a 16-core Windows desktop CPU, no GPU, 8 torch threads: about 1 minute from an empty cache (most of it is extracting features from about 3,900 images at roughly 100 to 150 images per second) and about 20 seconds with the cache filled. Expect two to four times longer on a typical student laptop, so plan for a few minutes. The downloaded weights are not counted.

The features are cached in `notebook/.cache/` (ignored by git), so a second run is much faster. Delete that folder to start over.

## The one download

The first run downloads the pretrained MobileNetV3-Large weights (about 21 MB) from PyTorch's servers into the torch cache (`~/.cache/torch/hub/checkpoints/` on macOS and Linux, `%USERPROFILE%\.cache\torch\hub\checkpoints\` on Windows). Nothing else touches the network, and after that the notebook works offline.

If you will be offline in class, run it once beforehand, or fetch the file by hand and copy it into that folder:

```text
https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth
```

If the download fails, the notebook prints this address and the target folder.
