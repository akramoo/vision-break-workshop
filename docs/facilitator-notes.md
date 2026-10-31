# Facilitator notes

Background concepts for running the workshop. **Educational only. Not for clinical use.**

**The project uses one model: MobileNet v2.** Everything else is either a dataset or a technique.

| Term | Model, dataset or technique? | Plain meaning |
|---|---|---|
| MobileNet v2 | **Model** (the only one) | A small neural network pre-trained by Google on 1.2 M everyday photos (ImageNet). We use it frozen: it turns each X-ray into 1,280 numbers. |
| Head | Model part (ours) | The tiny final layer we train on top of MobileNet's 1,280 numbers. It outputs P(pneumonia). |
| Kermany | Dataset | Children's chest X-rays (ages 1–5) from one hospital in Guangzhou, labeled normal or pneumonia. |
| ChestMNIST | **Dataset**, not a model | Part of the MedMNIST collection: NIH ChestX-ray14, 112,120 adult chest X-rays from the NIH Clinical Center (USA), resized. "224" is the image-size version (28, 64 and 128 also exist). We only take 400 adult images from it. |
| `Conv_1/Relu6` | **Layer name** inside MobileNet | MobileNet's last convolutional layer. Its output is a 7×7 grid (each cell sees a 32×32-pixel patch) of 1,280 feature maps. ReLU6 is its activation function, which clips values to 0–6. |
| Grad-CAM | **Technique** | Uses that 7×7 grid to colour the image regions that pushed the prediction, producing a heatmap. |

**The three test sets in the worst-case score.** A model is only as good as its worst set.

- **Kermany test:** the 624 X-rays (234 normal, 390 pneumonia) that the dataset authors set aside.
  - They come from the same hospital and age group as the training data, but the model never sees them during training.
  - It answers: *does it work on patients like the ones it learned from?*
  - Because 62.5% of the set is pneumonia, "always pneumonia" already scores 62.5%.
- **Adult test:** 200 adult X-rays from NIH (100 pneumonia, 100 with no finding).
  - The patients are adults, at another hospital, on other machines.
  - It answers: *does it still work on a different population?* This is distribution shift.
  - It is hand-balanced 50/50 because pneumonia is rare in NIH (about 1.2%), so "always pneumonia" scores 50% here.
  - Caveat: NIH labels were extracted from radiology reports by text software, so some are wrong.
- **Kermany stress:** the same 624 Kermany test images, each degraded in one fixed way.
  - The degradation is darker, lower contrast, rotated 10° and cropped to 85%, like a poor scan or a badly positioned patient.
  - The patients and labels are the same. Only the image quality changes.
  - It answers: *does it survive image-quality changes?* The fix is augmentation.
  - The preset is fixed so every student's leaderboard score is comparable.

**Is the split the default one?** Yes, it's the official split published by the dataset authors (Kermany et al. 2018): train 5,216 (1,341 normal / 3,875 pneumonia), validation 16 (8 / 8) and test 624. What we do with it:

- **Test:** the official test set, unchanged, so our numbers (including the 62.5%) match the literature.
- **Training subset:** 1,200 images (300 normal + 900 pneumonia), randomly sampled from the official train set.
  - It keeps the realistic ~1:3 ratio.
  - It downloads and processes fast on student laptops.
  - A frozen-feature model doesn't need 5,000 images.
- **Our own validation set:** 15% of that subset (180 images), "stratified", meaning it keeps the same 1:3 ratio as the rest.
- **The official 16-image validation set** is used only in the "dishonest evaluation" demo. With 16 images, one image moves the score by 6.25 points, so scores jump between runs.
- **Caveat:** the Hugging Face copy has no patient IDs. Our 180-image validation set may contain a second X-ray of a child who is also in training, which makes validation slightly optimistic. The official test set is separate.

**What `images [-1, 224, 224, 3]` means.** It describes the "tensor" (multi-dimensional array of numbers) that MobileNet accepts:

- `-1`: how many images in one batch. −1 means "any number".
- `224, 224`: height and width in pixels. This is MobileNet's standard size, and the data scripts resize every X-ray to it.
- `3`: colour channels (red, green, blue). MobileNet learned on colour photos, so it expects 3 channels. X-rays are grayscale, so the same gray value is copied into R, G and B.
- **Pixel values:** 0–255 in the JPEG, divided by 255, so the model receives numbers between 0 and 1.
- One X-ray is therefore 224 × 224 × 3 = 150,528 numbers (concept 3, "an image is numbers").

**Why this head (a single dense layer with a sigmoid, which is logistic regression)?**

- **Accuracy:** MobileNet's 1,280 features already do the hard work. A linear "probe" on frozen features is the standard transfer-learning baseline.
- **Overfitting:** with about 1,000 training images, fewer parameters (1,281) means less memorising.
- **Speed:** it trains in seconds even on weak laptops, so the live loss chart finishes during the explanation.
- **Reproducibility:** with a fixed seed, all 30 laptops get nearly the same numbers, so the class can compare.
- **Teaching:** it is the same model as the Python notebook's logistic regression, so both tracks compare directly. A single output, P(pneumonia), keeps the threshold, "refer to a doctor" and Grad-CAM simple.
- **Settings:** Adam with learning rate 1e-3 and batch 32 are the standard defaults. 20 epochs is enough to converge on about 1,000 examples, and the chart will show it.
- **Fallback:** if the dry run shows accuracy far below about 85%, I'll add an optional hidden layer under Advanced.

**Purpose of the "precomputed features" switch.** Running MobileNet over every image is the slow part, and it gives the same result every time because MobileNet is frozen.

- I compute those 1,280-number summaries once on my machine and ship them as small files (base: 5.7 MB).
- Students' laptops then only train the tiny head, which takes seconds instead of minutes.
- They also skip downloading all ~2,240 images.
- The switch defaults to ON. Turning it OFF shows students the real extraction speed on their laptop.

**Purpose of the shared pixel-ops file.** The same image steps run in two places: my precompute script (once, ahead of time) and the browser app (live, for uploads, webcam, sliders and custom stamps). The steps are scaling to 0–1, drawing the "R" stamp, the perturbations, the augmentation and the stress preset.

- If the two copies differed even slightly, the precomputed numbers wouldn't match what the app computes live.
- Students in precomputed mode would then get different results from students in live mode, and the stamp experiment could silently break.
- One shared file means one source of truth.

**What the "20-minute timebox" means.** It's a time limit I set for myself, not an install time.

- The precompute script uses `tfjs-node`, which contains compiled native code. Its install normally takes 1–2 minutes, but on Windows it sometimes fails.
- If I can't get it working within 20 minutes, I stop and switch to plan B: a hidden page in the app (`?tool=precompute`) that computes the same files in the browser.
- A tooling problem then can't eat the day.

## F1–F9 in plain words

| # | What the student sees and does | Workshop moment |
|---|---|---|
| F1 Data | Browse the X-rays and see the counts (3 pneumonia for every normal). Choose **realistic** or **balanced** training. On Day 2, switch on the fixes "add adult X-rays" and "augmentation". | Day 1 setup; Day 2 fixes |
| F2 Train | Click Train and watch the error drop and accuracy rise on a live chart. Advanced has epochs (passes over the data), learning rate (step size) and seed. "Repeat ×5" shows how much a 16-image validation set jumps around. | Day 1 Train v1; dishonest-evaluation break |
| F3 Evaluate | Scores on a test set: accuracy, **sensitivity** (sick children caught), **specificity** (healthy children cleared), the confusion matrix (4 boxes: right and wrong per class), and the 62.5% "always pneumonia" line beside them. | Day 1 concepts 8–9 |
| F4 Test sets | Choose what to test on: Kermany test, adult test, stress, your own uploaded pictures, or the live webcam. Point it at a cup and it still says "pneumonia" or "normal". | Closed-world break; Day 2 shift lab |
| F5 Perturb | Sliders for brightness, contrast, rotation and crop on the test images, with scores updating live. | Image-quality break |
| F6 Stamp | Put a white "R" marker on all **pneumonia** training images and retrain. Then stamp **normal** test X-rays and watch them flip to pneumonia. The flip rate measures it, before and after. | Day 2 shortcut lab |
| F7 Abstain | A confidence threshold. Below it, the app says **"Refer to a doctor"** instead of guessing. | Human in the loop |
| F8 Failure log | Record each failure: the image, the prediction, how sure the model was, a tag (why it failed) and a hypothesis. Export as a file. | Day 1 debrief taxonomy |
| F9 Results | A results card like the printed handout, with before- and after-fix columns and the **worst-case score** to post on the leaderboard. Also the precomputed switch and a Reset button. | Day 2 Fix it |

## Data caveats worth knowing

- The official Kermany test set contains 6 pixel-identical duplicate images. This comes from the source dataset, not from our preparation, and there is no overlap with the training data.
- Kermany images are used under CC BY 4.0 (Kermany et al. 2018, Mendeley Data, doi:10.17632/rscbjbr9sj.3). Full citations: DATA_LICENSES.md.
