# Dry run results

Educational only. Not for clinical use.

Date: 2026-10-31. Written by `app/tests/e2e/measure.spec.ts` (run with `MEASURE=1 npx playwright test --project=chromium-gpu --workers=1 measure.spec.ts`), on the production build served by `vite preview`.
The numbers below were produced by driving the real UI (clicks, tab switches, sliders) in the browser, reading the values the app shows on screen or in its debug state. Nothing was computed outside the app, except the spread statistics, which are taken over numbers the app produced.

## At a glance

- GPU proof: Chrome 156.0.8078.79 reports "ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, D3D11)"; the default headless Chromium reports SwiftShader.
- Day 1: cold load 5.0 s, train v1 3.6 s (validation 98.9%), Kermany test 82.4% accuracy / 99.5% sensitivity / 53.8% specificity against the 62.5% baseline; adult test 51.5%; stress 70.4%.
- Realistic model leans toward pneumonia: it flags 46.2% of healthy test children; class weights bring that to 38.5% (sensitivity 99.5% to 98.2%).
- Closed world: all six non-X-ray pictures are called pneumonia, 4 of them at 90 percent or more; a 0.9 threshold refers 2 of 6.
- Image quality: a slider update on 100 images takes 1.6 s at worst (target 5 s); brightness 0.4, rotation 20 degrees and a 70% crop drop balanced accuracy from 71.6% to 54.1%, 52.7% and 50%.
- Day 2: the default stamp flips 93.6% (219 of 234) of normal test images (smallest size reaching 50 percent: 32 px); the stamp-trained model's sensitivity on clean images is 58.2%; the adult drop is 30.9 points; worst-case goes 51.5% to 59.5% with the adult mix and the adult test is the worst set every time.
- Dishonest evaluation: over 20 seeds the 16-image score spans 6.3 points, the 180-image score 0.6 and the test score 3.5.
- Live extraction: 17.5 s for 1,824 images on the GPU, 3.6 min on software WebGL, so live mode is a usable fallback on both. The memory soak is flat in tensors and bytes.
- Compute fits every agenda block with a wide margin: largest share 3.6% on the GPU, 21% on software WebGL.

## Machine and browsers

| Item | Value |
| --- | --- |
| Computer | AMD Ryzen 7 5700X 8-Core Processor, 32 GB RAM, Windows 11 (build 26200), a desktop PC |
| GPU | NVIDIA GeForce RTX 3070, 8 GB video memory (nvidia-smi reports 8192 MiB), driver 617.42 |
| GPU browser | Google Chrome 156.0.8078.79 (installed build, headless, ANGLE on Direct3D 11 forced on) |
| Software-WebGL reference | Playwright bundled Chromium 156.0.8078.4 (headless, SwiftShader) |
| Other engine | Firefox 157.0 (Playwright build) |
| Seeds | Train seed 42 (the default); Repeat x5 uses seeds 1 to 5; the 20-seed study uses seeds 1 to 20; the 100-image perturb sample is the fixed stratified sample the app uses |
| Training recipe | 20 epochs, learning rate 1e-3, batch 32, precomputed features unless a section says otherwise |

## Is the GPU really used?
Playwright's bundled headless Chromium renders WebGL in software, so timings from it say little about a real laptop. The renderer string below is read inside each browser with `WEBGL_debug_renderer_info` (unmasked), and again from the exact WebGL context TensorFlow.js computes on.

| Project | Browser | WebGL renderer (unmasked) | TensorFlow.js context | Header badge |
| --- | --- | --- | --- | --- |
| chromium-gpu | chromium 156.0.8078.79 | ANGLE (NVIDIA, NVIDIA GeForce RTX 3070 (0x00002484) Direct3D11 vs_5_0 ps_5_0, D3D11) | same context | WebGL |
| chromium | chromium 156.0.8078.4 | ANGLE (Google, Vulkan 1.3.0 (SwiftShader Device (Subzero) (0x0000C0DE)), SwiftShader driver) | same context | WebGL |
| firefox | firefox 157.0 | ANGLE (NVIDIA, NVIDIA GeForce GTX 980 Direct3D11 vs_5_0 ps_5_0), or similar | same context | WebGL |

Chrome reports the RTX 3070 through ANGLE and Direct3D 11, so the chromium-gpu numbers are real GPU numbers. The default headless Chromium reports SwiftShader, a CPU renderer. Firefox hides the exact card on purpose (its string is a generic "or similar" ANGLE Direct3D 11 name, here shown as a GTX 980); the timings below show it is hardware accelerated too: it matches Chrome on the GPU and is roughly ten to twenty times faster than SwiftShader.

## A. Day 1 timed path (GPU)

| Step | Wall time | Note |
| --- | --- | --- |
| Cold load to "Ready" (empty cache, MobileNet warm-up) | 5.05 s | backend badge WebGL |
| Data tab (counts and first thumbnails) | 0.06 s | Using precomputed features |
| Train v1 (20 epochs, precomputed, seed 42) | 3.63 s | val acc 98.9% |
| Evaluate Kermany test | 0.15 s | acc 82.4 sens 99.5 spec 53.8 |
| Evaluate Adult test | 0.15 s | acc 51.5 sens 90 spec 13 |
| Evaluate Kermany stress | 0.23 s | acc 70.4 sens 99.2 spec 22.2 |
| Train balanced (class weights, precomputed) | 1.61 s | recipe switch 240 ms |
| Evaluate the balanced model on all three sets | 0.41 s |  |
| Webcam: start to first prediction | 0.85 s | overlay 3.7 fps |
| Upload 4 images to predictions | 2.15 s |  |

Train v1 detail: 20 epochs, seed 42, realistic balance, precomputed features (yes). Training accuracy 99.0 percent, validation accuracy (our 180 images) **98.9 percent**. The app reports 3.57 s inside the head; the rest of the wall time is the click and the UI refresh.

### Evaluate, realistic training (the default model)

| Test set | Accuracy | Sensitivity | Specificity | Balanced accuracy | TN / FP / FN / TP | False alarms on normals | Always-pneumonia baseline (acc / sens / spec) | Time to show |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kermany test (624) | 82.4% | 99.5% | 53.8% | 76.7% | 126 / 108 / 2 / 388 | 46.2% | 62.5% / 100% / 0% | 0.15 s |
| Adult test (200) | 51.5% | 90% | 13% | 51.5% | 13 / 87 / 10 / 90 | 87.0% | 50% / 100% / 0% | 0.15 s |
| Kermany stress (624) | 70.4% | 99.2% | 22.2% | 60.7% | 52 / 182 / 3 / 387 | 77.8% | 62.5% / 100% / 0% | 0.23 s |

### Realistic versus balanced (class weights)

| Test set | Model | Accuracy | Sensitivity | Specificity | Balanced accuracy | False alarms on normals | Said "pneumonia" overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kermany test | realistic | 82.4% | 99.5% | 53.8% | 76.7% | 46.2% | 79.5% |
| Kermany test | balanced | 84.5% | 98.2% | 61.5% | 79.9% | 38.5% | 75.8% |
| Adult test | realistic | 51.5% | 90% | 13% | 51.5% | 87.0% | 88.5% |
| Adult test | balanced | 51.5% | 87% | 16% | 51.5% | 84.0% | 85.5% |
| Kermany stress | realistic | 70.4% | 99.2% | 22.2% | 60.7% | 77.8% | 91.2% |
| Kermany stress | balanced | 73.4% | 99% | 30.8% | 64.9% | 69.2% | 87.8% |

On the Kermany test set (62.5 percent of the images really are pneumonia) the realistic model says "pneumonia" for 79.5 percent, and 46.2 percent of the healthy children are flagged. So it does lean toward pneumonia: sensitivity 99.5 percent against specificity 53.8 percent. Class weights move it to sensitivity 98.2 percent and specificity 61.5 percent (+7.7 points specificity, -1.3 points sensitivity); the model now says "pneumonia" for 75.8 percent and flags 38.5 percent of the healthy children. The always-pneumonia baseline is exactly 62.5 percent accuracy, 100 percent sensitivity, 0 percent specificity, as the plan says.

Webcam (Playwright fake device): first prediction 0.85 s after pressing Start; the overlay updated 22 times in 6.0 s = **3.7 frames per second** (the app throttles to about 4).

## B. Closed world: pictures that are not X-rays

Eight generated pictures (the six asked for plus a flat black and a flat white one) were uploaded on Break it, Your images. "Confidence" is the "N percent sure" the app shows for the class it picked.

### Default (realistic) model

| Picture | Predicted class | Confidence | Referred at 80% | Referred at 90% |
| --- | --- | --- | --- | --- |
| photo-scene | pneumonia | 93% | no | no |
| colour-gradient | pneumonia | 100% | no | no |
| random-noise | pneumonia | 79% | yes | yes |
| drawn-cup | pneumonia | 83% | no | yes |
| solid-red | pneumonia | 100% | no | no |
| text-page | pneumonia | 100% | no | no |
| solid-black (extra) | pneumonia | 100% | no | no |
| solid-white (extra) | pneumonia | 100% | no | no |

Of the six pictures, 6 are called pneumonia and 0 normal; 4 come with at least 90 percent confidence. A threshold of 0.8 refers 1 of 6 and 0.9 refers 2 of 6 (all eight: 1 and 2). 4 of the six non-X-rays are still answered with a confident class at a 90 percent threshold, so the model is confidently wrong on most of them and the abstain threshold only catches the least certain pictures.

### Balanced model

| Picture | Predicted class | Confidence | Referred at 80% | Referred at 90% |
| --- | --- | --- | --- | --- |
| photo-scene | pneumonia | 95% | no | no |
| colour-gradient | pneumonia | 100% | no | no |
| random-noise | pneumonia | 88% | no | yes |
| drawn-cup | pneumonia | 85% | no | yes |
| solid-red | pneumonia | 100% | no | no |
| text-page | pneumonia | 100% | no | no |
| solid-black (extra) | pneumonia | 100% | no | no |
| solid-white (extra) | pneumonia | 100% | no | no |

Of the six pictures, 6 are called pneumonia and 0 normal; 4 come with at least 90 percent confidence. A threshold of 0.8 refers 0 of 6 and 0.9 refers 2 of 6 (all eight: 0 and 2). 4 of the six non-X-rays are still answered with a confident class at a 90 percent threshold, so the model is confidently wrong on most of them and the abstain threshold only catches the least certain pictures.

Verdict on the closed-world claim: **it holds.** The default model answers "pneumonia" for all 6 pictures, and 4 of them with at least 90 percent confidence (a flat red square and a page of text get 100 percent). The model has no "none of the above" answer, so every picture gets a class. The abstain threshold helps only a little: at 0.8 it refers 1 of 6, at 0.9 it refers 2 of 6. Only the noise and the cup are caught, and raising the threshold that far also refers real X-rays (the final fixed model at 0.9 refers 174 (27.9%) of the Kermany test images).

## C. Image quality sliders (100-image Kermany test sample, default model)

The sample is the app's stratified 100 images (63.0% pneumonia, so the always-pneumonia baseline is that accuracy). Unperturbed, the default model scores 71.6 percent balanced accuracy (79 percent accuracy) on it. First run (loads the 100 images and predicts them twice): Computed in 2.8 s for 100 images. (5.19 s wall from opening the tab). "Update" is the app's own reported compute time for one slider move; "Wall" is the time from the slider change to the finished table and includes the 300 ms debounce.

| Setting | Balanced accuracy | Change vs original | Predictions changed (of 100) | Mean absolute confidence change | Update | Wall |
| --- | --- | --- | --- | --- | --- | --- |
| Brightness -0.1 | 71.6% | +0.0 | 0 of 100 | 2.3% | 0.3 s | 0.6 s |
| Brightness +0.1 | 70.8% | -0.8 | 1 of 100 | 1.4% | 0.2 s | 0.5 s |
| Brightness -0.2 | 68.9% | -2.7 | 4 of 100 | 4.4% | 0.2 s | 0.5 s |
| Brightness +0.2 | 66.8% | -4.8 | 4 of 100 | 5% | 0.2 s | 0.5 s |
| Brightness -0.4 | 61.4% | -10.2 | 8 of 100 | 8.8% | 0.2 s | 0.5 s |
| Brightness +0.4 | 54.1% | -17.5 | 13 of 100 | 12.8% | 0.2 s | 0.5 s |
| Contrast 0.7 | 68.9% | -2.7 | 2 of 100 | 4.2% | 0.5 s | 0.9 s |
| Contrast 0.85 | 70.3% | -1.3 | 1 of 100 | 2.1% | 0.2 s | 0.5 s |
| Contrast 1.3 | 72.2% | +0.6 | 2 of 100 | 4.6% | 0.2 s | 0.5 s |
| Rotation 3 deg | 68.1% | -3.5 | 5 of 100 | 5% | 1.6 s | 2.0 s |
| Rotation 5 deg | 64.1% | -7.5 | 8 of 100 | 6.5% | 0.2 s | 0.6 s |
| Rotation 10 deg | 60.8% | -10.8 | 10 of 100 | 10.7% | 0.2 s | 0.6 s |
| Rotation 20 deg | 52.7% | -18.9 | 16 of 100 | 15.3% | 0.2 s | 0.5 s |
| Crop 95% | 70.3% | -1.3 | 3 of 100 | 5.2% | 0.4 s | 0.7 s |
| Crop 85% | 59.5% | -12.1 | 11 of 100 | 12.1% | 0.2 s | 0.5 s |
| Crop 70% | 50% | -21.6 | 16 of 100 | 17.4% | 0.2 s | 0.5 s |

Compute time per slider update on the GPU: median 0.2 s, slowest 1.6 s (the app's reported time); wall 0.5 s to 2.0 s. The target was 5 s or less for 100 images: **met with a wide margin**. The slow outliers are the first move of each kind of operation (the GPU compiles a new shader program), not the data. Setting everything back to identity reproduced the original predictions exactly (0 of 100 changed).

Reading: the small changes the plan calls "small" (brightness 0.1, contrast 0.85, rotation 3 degrees, crop 95 percent) change 0 to 5 of 100 predictions and move the mean confidence by 1.4 to 5.2 percent. The big ones do real damage: brightness +0.4 takes balanced accuracy from 71.6 to 54.1 percent, rotation 20 degrees to 52.7 percent and a 70 percent crop to 50 percent (50 percent is a coin flip). Confidence swings on small changes are visible in the mean change column, but few predictions actually flip; the dramatic demos are the large settings.

## D. Day 2

### D1. The stamp experiment (default stamp: 48 px, top left, on pneumonia training images)

| Model | True class of the test images | Said "pneumonia" without stamp | Said "pneumonia" with stamp | Flip rate |
| --- | --- | --- | --- | --- |
| Before: normal model | Normal (234) | 46.2% | 63.7% | 17.5% (41 of 234) |
| Before: normal model | Pneumonia (390) | 99.5% | 100% | 0.5% (2 of 390) |
| After: stamp-trained model | Normal (234) | 1.3% | 94.9% | 93.6% (219 of 234) |
| After: stamp-trained model | Pneumonia (390) | 58.2% | 100% | 41.8% (163 of 390) |

Retraining on the stamped set plus computing the flip table took 1.8 s wall (Trained 20 epochs in 1.2 s.); re-computing the flip table alone took 0.05 s. On clean (unstamped) Kermany test images the stamp-trained model has accuracy 73.4 percent, **sensitivity 58.2 percent**, specificity 98.7 percent (the normal model: 82.4, 99.5, 53.8). A model that learned the marker misses 41.8 percent of real pneumonia when the marker is absent and flips 93.6% (219 of 234) of normals when it is added.

### Smallest stamp that still fools the model

Each row retrains the stamp-trained model with that stamp size (stamp on pneumonia training images, live feature extraction because the size is not the precomputed one) and then stamps the normal test images. Time is the whole retrain plus the Kermany flip measurement. The 48 px row is the default stamp, which uses the precomputed stamped features, so it takes about two seconds; every other size extracts its 765 stamped training images and 234 stamped test images live.

| Stamp size | Normal test images flipped to pneumonia, stamp-trained model (Kermany, 234) | Same for the unchanged normal model | Adult "no finding" images flipped (100) | Sensitivity on clean test images | Time for this size |
| --- | --- | --- | --- | --- | --- |
| 8 px | 6.4% (15 of 234) | 1.3% (3 of 234) | 5.0% (5 of 100) | 99% | 13.8 s (+ 1.0 s adult) |
| 12 px | 10.3% (24 of 234) | 2.6% (6 of 234) | 8.0% (8 of 100) | 98.5% | 8.0 s (+ 1.0 s adult) |
| 16 px | 12.8% (30 of 234) | 3.4% (8 of 234) | 11.0% (11 of 100) | 97.7% | 11.8 s (+ 1.6 s adult) |
| 24 px | 27.8% (65 of 234) | 5.6% (13 of 234) | 13.0% (13 of 100) | 96.9% | 11.1 s (+ 0.9 s adult) |
| 32 px | 53.4% (125 of 234) | 13.7% (32 of 234) | 15.0% (15 of 100) | 91.8% | 9.8 s (+ 1.0 s adult) |
| 48 px | 93.6% (219 of 234) | 17.5% (41 of 234) | 36.0% (36 of 100) | 58.2% | 2.1 s (+ 0.1 s adult) |

Smallest stamp with a flip rate of at least 50 percent on the Kermany normals: **32 px (53.4% (125 of 234))**. On the 100 adult "no finding" images no tested size reaches 50 percent, because the shortcut is weaker on images from another population. For the live challenge on the day this means a stamp has to cover a clear patch of the 224-pixel picture (about one seventh of its width) before the shortcut dominates; below roughly 16 px the effect is small.

### Population shift: the adult drop

| Model | Kermany test accuracy | Adult test accuracy | Drop (points) | Adult sensitivity | Adult specificity |
| --- | --- | --- | --- | --- | --- |
| Normal model | 82.4% | 51.5% | -30.9 | 90% | 13% |
| Stamp-trained model (clean images) | 73.4% | 56% | -17.4 | 76% | 36% |

The adult set is hand-balanced 50/50, so "always pneumonia" scores 50 percent there; the normal model's 51.5 percent is 1.5 points above that. Almost all of the failure is false alarms: adult specificity is 13 percent. The drop (30.9 points) is clear and easy to explain.

### The fix sequence and the worst-case score

Each row is a fresh training run with the recipe from the previous row plus one change, followed by "Evaluate all three sets" on the Results tab. The last two rows only change the refer threshold, which re-scores the stored predictions with half credit for referred cases. The worst-case score is the lowest of the three sets' scores; the set that sets it is named.

| Step | Train | Evaluate all | Kermany test | Adult test | Kermany stress | Worst-case | Worst set | Marked as worst in the card |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (realistic, tau 50%) | 2.70 s | 0.14 s | 76.7% | 51.5% | 60.7% | **51.5%** | Adult test | Adult test |
| + balanced class weights | 1.83 s | 0.05 s | 79.9% | 51.5% | 64.9% | **51.5%** | Adult test | Adult test |
| + adult X-rays mixed in | 3.13 s | 0.05 s | 79.2% | 59.5% | 70.1% | **59.5%** | Adult test | Adult test |
| + augmentation | 5.69 s | 0.04 s | 77% | 59% | 78.5% | **59%** | Adult test | Adult test |
| + refer below 80% (half credit) | - | - | 74.9% | 56% | 76.8% | **56%** | Adult test | Adult test |
| + refer below 90% (half credit) | - | - | 74.1% | 55.8% | 73.6% | **55.8%** | Adult test | Adult test |

Snapshots saved on the Results tab: "Before fix" (baseline), "After fix" (after balanced + adult mix + augmentation, threshold 80 percent) and "After fix (tau 90)" (threshold 90 percent). Leaderboard values: 51.5% Adult test | 56.0% Adult test +4.5 pts | 55.8% Adult test -0.3 pts. The app shows the delta of each snapshot against the previous one: +4.5 pts for "After fix" and -0.3 pts for "After fix (tau 90)". The "Before fix" to "After fix" change is therefore +4.5 pts.

Reading: the adult test is the worst set in every row, so the leaderboard is decided by the population shift, not by the Kermany sets. Class weights alone (51.5% to 51.5%) do nothing for the worst case; mixing in adult X-rays is the fix that moves it (to 59.5%); augmentation mainly repairs the stress set (70.1% to 78.5%) and leaves the worst case where it was. The refer threshold lowers the worst case here (59% to 56% at 80 percent and 55.8% at 90 percent), because the cases it refers are still right more than half of the time, so half credit is worth less than answering them. The accuracy columns of the same snapshots go the other way (accuracy on the answered cases rises when the threshold rises), which is easy to misread. The honest lesson for the "human in the loop" block is that a threshold trades coverage for safety, not for score.

## E. Dishonest evaluation: how much do scores jump between runs?

The app's "Repeat x5" (seeds 1 to 5) took 5.8 s. The same seeds were also run one by one with the normal Train button to extend the study to 20 seeds (1 to 20) in 66.0 s; for seeds 1 to 5 both routes gave identical test and 16-image accuracies (checked, identical). Accuracy in percent; the standard deviation uses n minus 1.

| Seeds | Evaluation set | Min | Max | Spread (points) | Std dev | Mean |
| --- | --- | --- | --- | --- | --- | --- |
| 5 (app) | Official validation, 16 images | 81.3 | 87.5 | 6.3 | 3.42 | 83.8 |
| 5 (app) | Our validation, 180 images | 98.9 | 99.4 | 0.6 | 0.30 | 99.2 |
| 5 (app) | Kermany test, 624 images | 79.2 | 81.3 | 2.1 | 0.87 | 80.7 |
| 20 | Official validation, 16 images | 81.3 | 87.5 | 6.3 | 2.56 | 86.3 |
| 20 | Our validation, 180 images | 98.9 | 99.4 | 0.6 | 0.28 | 99.2 |
| 20 | Kermany test, 624 images | 79.2 | 82.7 | 3.5 | 0.88 | 80.9 |

The app's own spread line after Repeat x5 reads: 16 images 6.3 points, 180 images 0.6 points, test 2.1 points. With 16 images one image is 6.25 points, so the 16-image score hops in steps of 6.25 (87.50, 81.25 percent were seen) while the 180-image score varies by about half a point. Picking the "best" of 20 seeds on the 16 images selects seed 1, whose honest test accuracy is 80.8 percent against a 20-seed mean of 80.9 percent. Note that the seed 42 model shown to students (82.4 percent on the test set) sits at the top of the 20-seed test range (79.2 to 82.7): a fresh seed will usually score 80 to 81, so say "about 81" on stage.

## F. The plan's assumptions

| Assumption | Result | Verdict |
| --- | --- | --- |
| 1. Frozen MobileNet features reach about 85 to 86 percent (ResNet-18 reference) | Kermany test accuracy 82.4% (seed 42); 20-seed mean 80.9% (range 79.2 to 82.7). Balanced model 84.5%. Specificity is the weak side (53.8%). | Below the reference by 3 to 6 points; show the gap honestly as planned. |
| 2. Live feature extraction is viable as a fallback | GPU: 1824 images (train 1,020 + validation 180 + Kermany test 624) took 17.5 s in total (train from scratch 13.0 s, Kermany evaluation 4.5 s); results identical to precomputed (82.4% vs 82.4%). Software WebGL: 3.6 min for the same work (train 2.4 min, evaluation 72.4 s). | Viable on a GPU laptop (well under a minute); on a laptop without GPU acceleration it is a few minutes, still usable as a fallback if students wait while it runs. The Data tab's own estimate ("Live mode: about 1200 images still need to be processed by MobileNet, roughly 2 min 24 s on this laptop. Turn on "Use precomputed features" in Settings to skip this.") is far too pessimistic for a GPU. |
| 3. The stamp flips normal X-rays reliably | Default 48 px stamp: 93.6% (219 of 234) of normal test images flip to pneumonia (the normal model: 17.5% (41 of 234)); on adult "no finding" images the flip rate is 36.0% (36 of 100). | Yes on the Kermany test set, weaker on adult images. |
| 4. The adult test shows a clear drop | 82.4% to 51.5% accuracy (-30.9 points); adult specificity 13%, barely above the 50% baseline overall. | Yes, clear. |
| 5. 16-image validation swings | Spread over 20 seeds: 6.3 points on 16 images against 0.6 on 180 and 3.5 on the test set. | Yes, the 16-image set swings by 6.25 points (one image) between seeds. |
| 6. Memory soak: 10 cycles of train, evaluate, perturb (100), stamp toggle | Tensors 106 at start and 106, 106, 106, 106, 106, 106, 106, 106, 106, 106 after each cycle; tf.memory().numBytes 8827272 at start, 8827272 to 8827272 during the cycles (GPU bytes tracked by tf: 9152544); JS heap after a forced garbage collection: first cycle 108.7 MB, last 159.9 MB (growth 51.2 MB; mean of the first half 151.1 MB, second half 159.8 MB). | Tensor count and tf bytes are exactly flat. The JS heap steps up once between cycle 1 and cycle 2 (108.7 to 159.1 MB, probably caches filling) and then shows no trend: cycle 2 159.1 MB, cycle 10 159.9 MB, readings between 150.3 and 169.4 MB. |

## GPU Chrome, software-WebGL Chromium and Firefox

| Measure | Chrome with GPU | Chromium, software WebGL | Firefox |
| --- | --- | --- | --- |
| Renderer | RTX 3070 (ANGLE D3D11) | SwiftShader (CPU) | ANGLE D3D11 (card name masked) |
| Cold load to Ready | 3.3 s | 2.1 s | 4.3 s |
| Train 20 epochs (precomputed) | 2.79 s | 4.34 s | 3.08 s |
| Show Kermany test results | 0.12 s | 0.15 s | 0.10 s |
| Perturb 100 images, first run (as reported by the app) | Computed in 4.2 s for 100 images. | Computed in 12.7 s for 100 images. | Computed in 4.7 s for 100 images. |
| Perturb 100 images, brightness update | 0.5 s | 10.5 s | 0.5 s |
| Perturb 100 images, rotation update | 1.8 s | 11.3 s | 1.7 s |
| Live extraction speed (200 adult images) | 52 images/s | 8.6 images/s | 51 images/s |
| Kermany test accuracy (must match) | 82.4% | 82.4% | 82.4% |

The three browsers give the same accuracy to the digit (the model code is identical), so only speed differs. A laptop with a working GPU behaves like the first or third column; an old or virtual machine without WebGL acceleration behaves like the middle one.

## Workshop timing

Compute time is the wall time the app needs for everything a student does in that block, measured on the GPU; the column "Without a GPU" is an estimate for a laptop with software WebGL, scaled from the measured ratios between the two browsers (training about the same, perturbation and live extraction much slower). The time budget is the agenda block length. Students also read, discuss and log, so the compute is only part of each block.

| Agenda block | Budget | What the app computes | Compute (GPU) | Share of budget (GPU) | Compute (no GPU, estimate) | Share of budget (no GPU) |
| --- | --- | --- | --- | --- | --- | --- |
| Day 1 Setup (open the app, load the data) | 10 min | cold load to Ready, Data tab, one webcam start, one 4-image upload | 8.1 s | 1.4% | 13.3 s | 2% |
| Day 1 Train v1 | 25 min | train (precomputed), evaluate the three sets, Repeat x5, train balanced and compare | 12.0 s | 0.8% | 18.6 s | 1% |
| Day 1 Break round 1 | 25 min | first perturb run plus 20 slider moves, a webcam session, an upload of 4 images | 21.4 s | 1.4% | 4.0 min | 16% |
| Day 2 Shortcut lab | 30 min | train, stamp-train with the default stamp and flip rate, then the six-size challenge (live extraction, kermany + adult) | 65.1 s | 3.6% | 6.3 min | 21% |
| Day 2 New population | 20 min | adult test evaluation and the adult flip table (all precomputed) | 0.2 s | 0.0% | 0.4 s | 0% |
| Day 2 Fix it | 25 min | four trainings (baseline, balanced, adult mix, augmentation) with "Evaluate all" after each, plus two threshold changes | 13.6 s | 0.9% | 21.2 s | 1% |

Conclusion: on a GPU laptop the largest block uses 3.6 percent of its time budget in compute, so the compute fits with a very wide margin and the time goes to teaching. On a laptop without a GPU the largest share is 21 percent (still fits, but the perturbation sliders and the stamp-size challenge will feel slow, with 10.5 s per slider move). Loading the app and the model is the same for everyone and takes a few seconds on a local server.

Perturbation scale factor between software and GPU: about 21x. Live extraction scale factor: about 6.0x.

## Observations from this run

1. **The normal model also reacts to the stamp.** With the default stamp on normal test images the unchanged normal model flips 17.5% (41 of 234), not "barely" as the note under the flip table says; the stamp-trained model flips 93.6% (219 of 234). The difference between the two, not the absolute number, is the lesson. (Low, wording.)
2. **Class weights do not move the worst-case score; the abstain threshold lowers it.** The worst-case set is the adult test in every step. Balanced weights leave it at 51.5%, the adult mix lifts it to 59.5%, and a refer threshold of 80 or 90 percent lowers it again (half credit is worth less than the accuracy the model has on the cases it refers). The agenda lists the threshold as a "fix"; tell students it buys safety, not score. (Medium, workshop content.)
3. **The Data tab live-mode estimate is calibrated for a laptop without a GPU.** It says about 2 min 24 s for 1,200 images; this PC needs 13.0 s for the whole live training, and software WebGL needs 2.4 min. The estimate is right for the slow case and about ten times too high for a GPU. (Low.)
4. **The 82.4 percent test accuracy of seed 42 is at the top of its range.** Over 20 seeds the Kermany test accuracy is 79.2 to 82.7 percent (mean 80.9). The facilitator notes say "about 82"; "about 81, between 79 and 83" is safer if students pick other seeds. (Low.)
5. **The first move of each slider kind is slower** (rotation 3 degrees took 1.6 s while later rotations took 0.2 s) because the GPU compiles a new shader program. Harmless; mention it if someone asks why the first rotation lags. (Info.)
6. **Firefox is hardware accelerated here too.** It hides the card name, but its timings match Chrome on the GPU (perturb update 0.5 s against 0.5 s), so Playwright's Firefox is not a software reference; only the bundled headless Chromium is. (Info.)
7. **The machine has 8 GB of video memory** (nvidia-smi), not 4 GB; this does not change any conclusion because tfjs holds only about 9 MB on the GPU during the soak.

No flow of the click path was blocked or crashed, and the shared test fixture, which fails any test that requests a non-localhost origin, never fired.

Screenshots from this run: `docs/screenshots/dryrun-evaluate-kermany.png`, `dryrun-closed-world.png`, `dryrun-perturb.png`, `dryrun-stamp-flip.png`, `dryrun-results-snapshots.png`.

