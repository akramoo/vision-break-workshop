# Build a Vision Model, Then Break It

A teaching app for a two-day university workshop (1 and 2 November 2026). Students train a pneumonia detector on children's chest X-rays, entirely in the browser, and then spend most of the session finding out how it fails.

**Educational only. Not for clinical use.** The X-rays come from published, anonymised research datasets. Nothing you load or capture leaves your laptop.

Live app: <https://akramoo.github.io/vision-break-workshop/>

## The idea

Training a classifier that scores well on a test set takes about ten minutes. Knowing whether that score means anything takes the rest of the workshop. The model here is a frozen MobileNet v2 that turns each X-ray into 1,280 numbers, plus a small trainable layer on top that predicts pneumonia or normal. It trains in seconds, even on a weak laptop, so nobody waits and everybody has time to poke at it.

The first lesson is the baseline. About 62.5% of the official Kermany test images show pneumonia, so a model that always answers "pneumonia" already scores 62.5%. Every number the app shows is placed next to that.

Then students break their model in six ways:

| Break | What students do | Why it happens |
| --- | --- | --- |
| Imbalance | Compare the model with "always pneumonia"; switch between 1:3 and balanced training | Training data has three pneumonia images for every normal one |
| Shortcut | Stamp a small white "R" marker on pneumonia training images only, retrain, stamp a normal X-ray | The model learns the marker instead of the lungs |
| Population shift | Test on adult X-rays from the NIH dataset | Children from one hospital are not everyone |
| Image quality | Move brightness, contrast, rotation and crop sliders | Small changes in the picture move the confidence a lot |
| Closed world | Point the webcam at a cup | The model must pick a class, so it picks one, confidently |
| Dishonest evaluation | Pick a model using the official 16-image validation set | One image moves the score by six points, so results jump between runs |

Day 2 ends with fixes: balance the classes, augment the training images, mix in adult X-rays, and let the model say "Refer to a doctor" when it is unsure. Each fix is measured on three test sets (Kermany test, adult test, and the Kermany test with a fixed image degradation) and the worst result is the score that goes on the leaderboard.

## Status

The app is being built one feature at a time and the live site is updated with each step.

- Done: data preparation (Kermany subset and MobileNet v2 files), app shell with WebGL, WASM and CPU backends, shared image code, Pages deployment.
- In progress: loading data and class counts, training with a live chart, evaluation, test sets, perturbations, the stamp tool, the abstain threshold, the failure log and the results card.
- Optional extras after that: Grad-CAM heatmaps and a Python notebook for students who finish early.

## Run it locally

You need Node 22 or newer.

```bash
cd app
npm ci
npm run dev
```

Open the address Vite prints. The camera only works on `https://` or `localhost`, which is why the hosted version is the one to use in class.

Other commands, all from `app/`:

```bash
npm run build        # type-check and build into app/dist
npm test             # unit tests (Vitest)
npm run e2e          # browser tests (Playwright, Chromium and Firefox)
```

The first time you run the browser tests, install the browsers with `npx playwright install chromium firefox`.

## Run it without internet

The built app has no external requests at all: the model, the images and the WebAssembly files are all part of `dist/`. Build once, copy the folder to a USB stick, and serve it from any laptop:

```bash
cd app
npm run build
node dist/serve-offline.mjs      # http://127.0.0.1:8080
```

`npx serve dist` and `python -m http.server` in the `dist` folder also work. Open it through `localhost` rather than double-clicking `index.html`, since browsers block model loading from `file://`.

## Rebuild the data

The prepared data is already in the repository, so you only need this if you want to regenerate it. You need Python 3.10 or newer and about 6 GB of free disk space for the downloads.

```bash
python -m venv scripts/.venv
scripts/.venv/Scripts/pip install -r scripts/requirements.txt    # Windows; use scripts/.venv/bin/pip elsewhere

node scripts/fetch_model.mjs                       # MobileNet v2 into app/public/models
scripts/.venv/Scripts/python scripts/prepare_kermany.py
```

`prepare_kermany.py` takes a seeded sample of 300 normal and 900 pneumonia images from the official training split, holds out 15% of it for validation, and keeps the full official test set (234 normal, 390 pneumonia). The same seed always produces the same files.

The scripts check their own output. To confirm a fresh copy is intact:

```bash
scripts/.venv/Scripts/python scripts/tests/verify_data.py
node scripts/tests/verify_model.mjs
```

## Deploying

The live site is served from the `gh-pages` branch, not from a GitHub Actions workflow, because Actions is not available on this account at the moment. One command builds the app and publishes it:

```bash
node scripts/deploy_pages.mjs
```

It runs `npm run build` in `app/`, pushes the contents of `app/dist` to `gh-pages`, and asks GitHub to rebuild Pages. The site is updated about a minute later. You need the GitHub CLI (`gh`) logged in with push access.

The workflow in `.github/workflows/pages.yml` is kept for the day Actions works again. It only runs when started by hand, and Pages would need its source switched back to "GitHub Actions".

The app uses relative paths throughout, so it works under any repository name.

## If something goes wrong in class

| Problem | What to do |
| --- | --- |
| Camera does not open | It needs `https://` or `localhost`. Use the hosted link. The webcam is only needed for the closed-world demo. |
| Training is slow | Keep "Use precomputed features" switched on. |
| No WebGL on a laptop | The app falls back to WebAssembly on its own. Adding `?backend=wasm` to the address forces it. |
| The tab gets sluggish after an hour | Use "Reset session" in the settings drawer. Results are saved locally first. |
| Internet drops | The app keeps working once it has loaded. Otherwise use the USB copy. |
| Everything breaks | Run it from your own laptop on the projector. |

## Workshop-day checklist

- Open the live link on the projector laptop in Chrome and in Firefox, and run one full train and evaluate.
- Allow the camera once on that laptop so the permission prompt is not a surprise.
- Copy `app/dist` to at least two USB sticks and test one on a different machine with Wi-Fi off.
- Have `?backend=wasm` ready in case a student laptop has no usable graphics.
- Reset the leaderboard before the second session and check that the adult test set loads.
- Say the disclaimer out loud at the start. It is also in the header and footer of every screen.

## Data and licenses

The code is MIT licensed. The datasets and the model keep their own licenses, with full citations, in [DATA_LICENSES.md](DATA_LICENSES.md). In short: the children's X-rays are Kermany et al. 2018 (CC BY 4.0), the adult X-rays are NIH ChestX-ray14 through MedMNIST v2 (CC0), and the feature extractor is MobileNet v2 (Apache 2.0).

Facilitator background (what each term and test set means, and why the pieces are the way they are) is in [docs/facilitator-notes.md](docs/facilitator-notes.md).
