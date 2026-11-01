# UI guide for the instructor

This guide is for the person standing at the front of the room. It explains what every tab, button and number in the app means, what to click and in what order, what students will see, and what can go wrong. It assumes you know the machine learning ideas (training, test sets, sensitivity, overfitting) but did not write the code.

All numbers below come from the default run: seed 42, 20 epochs, realistic 1:3 training, precomputed features on, on a laptop with no GPU. Your numbers should match to within a point or so. They are taken from the app's own screenshots in `docs/screenshots/` and from `docs/facilitator-notes.md`.

Educational only. Not for clinical use. Say it out loud on the first slide; the app also prints it in the header, the footer and the results card.

---

## 1. The app in one minute

### What it is

A single web page that trains a 	detector on children's chest X-rays, entirely inside the browser. Nothing is uploaded anywhere. The footer says so: "Runs entirely in your browser. X-rays and models never leave this computer."

The model has two parts:

| Part         | Frozen or trained?                          | What it does                                                                                                   |
| ------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| MobileNet v2 | Frozen, never changes                       | Turns each X-ray into 1,280 numbers. It was trained by Google on everyday photos.                              |
| The head     | Trained by the student, takes a few seconds | One layer (logistic regression) that turns the 1,280 numbers into P(pneumonia), a probability between 0 and 1. |

So when a student clicks Train, they are only fitting the head. That is why it takes about 3 seconds and why everyone can retrain many times.

### Where the data lives

- The X-ray images, the MobileNet model and the precomputed features are all files that ship with the page. There are no outside requests once it has loaded, so it keeps working if the Wi-Fi drops.
- The student's own work (trained model, failure log, snapshots, settings, checklist ticks) is stored in that browser on that laptop. It survives a page reload. It is wiped by Reset session.
- Uploaded pictures and webcam frames stay in memory only. They disappear on reload.

### The header

| Item                                  | What it shows                                                                                                                                                        | What to do about it                                                                                                                                                          |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Backend badge (left of the green one) | `WebGL`, `WASM`, `CPU`, `No backend`, or `Detecting…`. Hover for a tooltip. A yellowish badge means the app wanted WebGL, could not use it and fell back. | WebGL is the fast case. WASM is fine and still quick for this model. If a laptop shows CPU, expect slow Perturb and live extraction. Force a backend with `?backend=wasm`. |
| Model badge (green)                   | `Loading model… 40%` while MobileNet loads, then `Ready`. `Model unavailable` is the failure state, and a red banner explains it.                             | Nobody can train until it says Ready. Normally under a few seconds.                                                                                                          |
| Settings button                       | Opens the settings drawer (below).                                                                                                                                   |                                                                                                                                                                              |

### The tab bar

Six tabs, in the order you will use them: **Data, Train, Evaluate, Break it, Failure log, Results.** Arrow keys move between tabs. Work flows left to right, but nothing stops a student from jumping around.

### Workshop steps panel

Just under the tab bar there is a collapsed bar reading "Workshop steps (0 of 11 done)". Click it to open two columns of checkboxes, Day 1 and Day 2.

Day 1: Load data and read the class counts; Train your first model; Compare with 'always say pneumonia'; Evaluate on the adult X-rays; Break it three ways (sliders, upload, webcam on a cup); Log each failure with a tag.

Day 2: Run the stamp experiment and record the flip rate; Test on the adult set and note the drop; Apply two fixes (balance, adult mix, augmentation, refer threshold); Re-measure all three sets and save a snapshot; Post your worst-case score.

Ticking is manual. Nothing ticks itself. The ticks survive a reload. Use it as a "where am I" card for a stuck student, and as a way to read the room: "Hands up if you have ticked step 3."

### Settings drawer

Opens from the right. Press Escape or Close to leave.

- **Use precomputed features** (switch, on by default). On means the app reads MobileNet's 1,280 numbers for each image from files shipped with the page, so training and evaluation take seconds. Off means MobileNet is run on every image on this laptop. The result is the same numbers, but it takes minutes. You can flip it at any time; the next training or evaluation uses the new mode, and every result says which mode produced it.
- **Backend** and **Model SHA-256** are read-only info. The hash proves everyone has the same MobileNet file. You will never need it in class.
- **Reset session** (red button, asks for confirmation). Clears the trained model, the stamp-trained model, uploads, camera, failure log, snapshots, checklist ticks and saved settings. It also puts the precomputed switch back on, the refer threshold back to 50% and the training recipe back to defaults. MobileNet stays loaded, so it is quick. Use it for a stuck tab or a clean start. Do not use it casually, because it deletes the failure log and the snapshots.

### Footer

Repeats "Educational only. Not for clinical use." and the "runs entirely in your browser" line.

### Orange and red bars you may see

- A yellow notice bar with Dismiss: usually "Precomputed features could not be used, so features are being extracted live on this laptop (slower, same result)." Training still works, just slower.
- A red banner: "No compute backend available", "Model not available", or "WebGL context lost" (reload, or open with `?backend=wasm`).

---

## 2. Tab by tab

### 2.1 Data

**What students see.** A short sentence ("3 pneumonia X-rays for every normal one, a 1:3 ratio"), a green dot with "Using precomputed features", a table "How many images?", a "How should we train?" card, and a gallery.

**The count table.** Read it row by row:

| Set                           | Normal           | Pneumonia | Total | Used for                                 |
| ----------------------------- | ---------------- | --------- | ----- | ---------------------------------------- |
| Kermany train                 | 255              | 765       | 1020  | Training (children)                      |
| Kermany validation (ours)     | 45               | 135       | 180   | Progress check while training            |
| Kermany validation (official) | 8                | 8         | 16    | Only the "dishonest evaluation" demo     |
| Kermany test                  | 234              | 390       | 624   | The final exam. Never used for training. |
| Adult train pool              | 100 (no finding) | 100       | 200   | Optional fix: mix into training          |
| Adult test                    | 100 (no finding) | 100       | 200   | Shift test (adults, another hospital)    |

The line under the table reminds you that "Kermany stress" is not new data. It is the same 624 test X-rays, made darker, lower in contrast, rotated and cropped.

**The recipe card.**

- **Class balance:** Realistic (every image counts the same, so the common class dominates) or Balanced (mistakes on the rarer class count more). The line "Class weights used in this mode" shows normal 1.00 / pneumonia 1.00 for Realistic and, in grey, what Balanced would use (normal 2.00 / pneumonia 0.67). Plain meaning: in Balanced mode one normal X-ray counts three times as much as one pneumonia X-ray in the training error, which cancels the 1:3 imbalance.
- **Day-2 fixes:** two checkboxes, "Mix in adult X-rays (+200)" and "Augmentation (+2 views per image)". Leave both off on Day 1.
- **Training will use N rows:** 1020 rows (255 normal and 765 pneumonia) by default. With adult mix it is 1,220. With augmentation each image counts three times (original plus two random copies).
- If precomputed features are off, an orange line estimates how long live extraction will take.

**The gallery.** "Browse the X-rays" with two dropdowns (Set, Label) and Previous/Next. Thumbnails carry a green "normal" or red "pneumonia" chip; adult images show "no finding" under the Normal label.

**What to demo.**

1. Filter to Kermany train, normal, then pneumonia. Ask: "Can you tell them apart?" Most cannot.
2. Point at the 255 / 765 counts. Say "three to one" before anyone sees a score.
3. Filter to Adult test and flip through a page. Ask what looks different (larger body, different contrast, different machine).

**What to ask.** "If a model always said pneumonia, what accuracy would it get on the test set?" Let them compute 390 out of 624. Write 62.5% on the board.

**Gotchas.**

- Changing the recipe here does not change a model that is already trained. The Train tab's "Recipe" line shows the new settings immediately, but the existing model keeps the old ones until you press "Train again". Evaluate and Results describe the model that was actually trained.
- Adult "normal" images are called "no finding" everywhere in the app. They are adults whose report mentioned none of the 14 listed conditions. That is not the same claim as "healthy". The labels were mined from reports by software, so a few are wrong.
- Many of the children's X-rays carry a small letter marker (an "R" or "L") in a corner. These are natural side markers. Someone will ask whether they count as a shortcut. It is a good question: they appear on both classes, so they carry no information, which is exactly the point of the stamp lab.
- The training pool is 1,020 images plus 180 held out for validation, 1,200 in all. The facilitator notes say "1,200 training images"; the table above is what students will see.

### 2.2 Train

**What students see.** An intro sentence, a recipe card, two charts, an "Advanced settings" fold, and a "Repeat x5 with new seeds" card.

**Recipe card.**

- "Recipe (change it in the Data tab): realistic class balance. 1020 training rows."
- **Check progress on:** "Our validation set (180 images)" (default) or "Official validation set (16 images)". This only changes which images the validation line is drawn on. It never touches the test set.
- **Train** button (it becomes **Train again** after the first run) and **Cancel**. While running, a status line reads "Training: epoch 7 of 20, 1.2 s elapsed" and a progress bar fills. When done: "Finished 20 epochs in 3.0 s. Validation accuracy 98.9%."

**The two charts.**

| Chart                       | Reading it                                                                                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Loss (lower is better)      | The error the model is trying to shrink. Solid blue is training, dashed orange is validation. Both should fall from about 0.27 to about 0.05 and flatten. |
| Accuracy (higher is better) | Share correct. Both climb quickly to about 99% by epoch 8 or so.                                                                                          |

Both lines finishing near 99% is the trap. It looks like a great model. That score is on the validation set, which comes from the same pool of children as the training set. Tell students to remember "99%" because the next tab will take it away.

**Advanced settings (collapsed).**

- **Epochs** (passes over the training data), 20 by default.
- **Learning rate** (step size), slider, 1.0e-3 by default.
- **Seed** (same seed, same result), 42 by default, with a **New seed** button. Same seed and same settings give the same numbers on every laptop. Use this to get a classroom of identical results; change it to show that numbers move.

**Repeat x5 with new seeds.** Trains the same recipe five times with seeds 1 to 5 and shows a table:

| Run (seed)  | Validation accuracy, 16 images | Validation accuracy, 180 images | Kermany test accuracy |
| ----------- | ------------------------------ | ------------------------------- | --------------------- |
| Seed 1 to 5 | 81.3% to 87.5%                 | 98.9% to 99.4%                  | 79.2% to 81.3%        |
| Spread      | about 6.3 points               | about 0.6 points                | about 2.1 points      |

One image on the 16-image set is worth 6.25 points, so its score jumps. This is the "dishonest evaluation" demo: if you picked your best model using the official 16-image validation set, you would be choosing noise. It does not replace the student's model; it leaves the current head alone and throws its five models away.

**What to demo.**

1. Click Train with the projector on the charts. Talk over the 3 seconds about transfer learning: the big network is frozen, only the last layer learns.
2. When it finishes, say the validation accuracy aloud (about 99%) and do not comment on it yet.
3. Open Advanced, change the seed, retrain: the numbers move slightly. Change back to 42.
4. Later, Repeat x5 for the dishonest-evaluation break.

**What to ask.** "Is 99% good?" Expect "yes". Reply: "Compared with what?"

**Gotchas.**

- Training needs the Ready badge and the X-ray list loaded. Until then the button is disabled and a grey reason sits next to it.
- If a student switches to another browser tab, the browser slows the page's timers and training stretches out. Ask them to stay on the app.
- A model from a previous visit is restored automatically. The status then reads "A trained model from your last visit was restored. Train again to replace it." This is convenient but means a rehearsal laptop starts with a model already trained.
- The seed box, epochs and learning rate are disabled while a run is in progress.

### 2.3 Evaluate

**What students see.** "Test on" with four radio buttons: **Kermany test**, **Adult test**, **Kermany stress**, **Your uploaded images**. A one-line description of the chosen set. The "Refer to a doctor" slider. Then the results.

There is no Evaluate button. The scoring runs automatically the moment you open the tab with a trained model, and again whenever you pick a different set. With precomputed features it is nearly instant. If the student has not trained, the tab shows "No trained model yet. Go to the Train tab and click Train" with a button.

**The results block** for Kermany test, default model:

| Predictor              | Accuracy | Sensitivity | Specificity | Balanced accuracy |
| ---------------------- | -------- | ----------- | ----------- | ----------------- |
| Our model              | 82.4%    | 99.5%       | 53.8%       | 76.7%             |
| Always say "pneumonia" | 62.5%    | 100.0%      | 0.0%        | 50.0%             |

How to read it:

- **Accuracy:** share of all 624 X-rays called correctly. Compare it with the baseline row, never with 100%.
- **Sensitivity** (sick children caught): of the 390 children with pneumonia, 388 were flagged.
- **Specificity** (healthy children cleared): of the 234 healthy children, only 126 were cleared. The other 108 were false alarms.
- **Balanced accuracy:** the average of sensitivity and specificity. It is 50% for any model that ignores the picture, which makes it a fairer score than accuracy when the classes are unequal.
- **The baseline row** never looks at the picture. Its accuracy equals the share of pneumonia in the set.

**The confusion matrix.** Rows are the true class, columns what the model said. Green cells are right, red cells are wrong. Each cell shows a count, "x% of row" and a plain word.

|                          | Model said normal     | Model said pneumonia |
| ------------------------ | --------------------- | -------------------- |
| True normal (n = 234)    | 126 correctly cleared | 108 false alarm      |
| True pneumonia (n = 390) | 2 missed              | 388 caught           |

A blue note below explains that a missed sick child usually costs more than a false alarm.

**Individual predictions.** A table of the first 40 rows, "Show mistakes first" ticked by default. Each row has a thumbnail, true label, "Model says", a confidence, and a **Log this** button that opens a small form (tag, hypothesis, "Save to failure log"). Wrong answers are marked "(wrong)". The confidence is how sure the model is of the class it chose, between 50% and 100%.

**The three sets, with expected numbers (tau = 50%).**

| Set                  | What it tests                            | Accuracy | Sensitivity | Specificity | Balanced | Baseline |
| -------------------- | ---------------------------------------- | -------- | ----------- | ----------- | -------- | -------- |
| Kermany test (624)   | Same kind of patient as training         | 82.4%    | 99.5%       | 53.8%       | 76.7%    | 62.5%    |
| Adult test (200)     | New population (shift)                   | 51.5%    | 90.0%       | 13.0%       | 51.5%    | 50.0%    |
| Kermany stress (624) | Darker, lower contrast, rotated, cropped | 70.4%    | 99.2%       | 22.2%       | 60.7%    | 62.5%    |

On the adult test the table headers change to "sick patients caught" and "patients with no finding cleared", and the matrix reads "no finding" instead of "normal".

**Uploads option.** Choosing "Your uploaded images" swaps the results block for the upload manager (see 2.4). Only images the student has labelled count towards scores.

**The Refer to a doctor slider.** "Refer to a doctor when the model is less than X% sure." Details in the next section, because it appears here and in Break it.

**What to demo.**

1. Open Evaluate on Kermany test. Read the baseline row first. "62.5% for a model that does not look. Ours is 82.4%. Better, but look at specificity."
2. Point at the 108. "Our model cleared barely half the healthy children." Sensitivity is 99.5% because it leans hard toward answering pneumonia, which is what an imbalanced training set teaches.
3. Switch to Adult test. The accuracy drops to about 51.5%. The baseline there is 50%. The model is barely better than a coin.
4. Switch to Kermany stress. Accuracy is 70.4%, only eight points above the 62.5% baseline, and balanced accuracy is 60.7%, barely ten points above the 50% floor. Specificity has collapsed to 22.2%.

**What to ask.**

- "Which number would you quote to a hospital: 99%, 82% or 51%?" (The point is that 99% was on validation.)
- "Which mistake is worse here, a miss or a false alarm?"
- "Why does the adult test baseline say 50% when the Kermany baseline says 62.5%?" (Because we hand-picked 100 and 100. Real NIH data has about 1.2% pneumonia.)

**Gotchas.**

- Train before evaluating. Without a trained model the tab only shows the empty-state message.
- Training again discards every cached evaluation. Results and Evaluate recompute for the new model. Students who retrain right before the leaderboard step must press "Evaluate all three sets now" again.
- The precomputed switch matters here: a line under the table says "Features for this evaluation: precomputed (shipped with the app)" or "extracted live on this laptop". With the switch off, evaluating the 624 test images takes about a minute or more on a weak laptop.
- With the tau slider above 50%, Accuracy, Sensitivity and Specificity are computed only on the cases the model answered. Two new columns appear, Referred and Coverage. See 2.4.

### 2.4 The Refer to a doctor slider (shared control)

This slider sits on Evaluate and on Break it. It is one shared setting. Moving it in one place moves it in the other.

**Semantics, plainly.** The model always has a lean: P(pneumonia) at or above 50% means "pneumonia", below means "normal". Its confidence is the larger of P and 1 minus P, so between 50% and 100%. The slider sets a bar for confidence. If the model's confidence is below the bar, the app does not answer. It says "Refer to a doctor" and shows the case in orange.

- At 50% (the leftmost position, the default), nothing is ever referred. The label reads "50% = the model always answers (never refers)".
- Raising it to 90% means: "answer only if at least 90% sure". The range goes up to 99%.
- Raising it is not "making the model more accurate". It is making the model say less.

**Two new columns appear once the slider is above 50%.**

- **Referred:** how many cases, and what percentage of the set, were handed to a doctor.
- **Coverage:** the share of cases the model dared to answer. It is 100% minus the referred share.

At 90% on Kermany test with the default model: accuracy rises to 90.4%, 157 cases (25.2%) are referred and coverage is 74.8%. The confusion matrix title says "467 X-rays (157 referred, not counted)".

**Half credit, and why it exists.** Accuracy on only the answered cases looks better as the bar rises, because the model is picking its easy cases. To stop that from being a trick, the worst-case score on the Results tab gives half a point for every referred case. If a model refers everything, it scores exactly 50, the same as always saying pneumonia. In other words, you can not game the leaderboard by abstaining. On the same 90% setting, the Kermany half-credit score is 74.6%, which is lower than the 76.7% at 50%. The accuracy went up and the score went down.

**What to demo.** Evaluate on Kermany test, drag to 90%, watch accuracy jump to 90.4% and the Referred column fill. Then go to Results and show the half-credit score. Ask: "Is the model better, or just quieter?"

**Gotchas.**

- The slider is remembered across reloads. Reset session puts it back to 50%.
- The baseline row never refers anything, so it always shows a dash in those columns.
- On the adult set the model is so unsure of itself (or so confidently wrong) that raising the bar to 90% refers 34% of cases but leaves the half-credit score near 51.5%. The leaderboard value does not move. That is the right outcome: abstention cannot rescue a model that is wrong with high confidence.

### 2.5 Break it

A heading, a one-line lead, three sub-buttons (**Perturb**, **Stamp**, **Your images & webcam**), the shared tau slider, then the chosen experiment. All three need a trained model.

#### Perturb

**What students see.** A card with "Test images" (Kermany test, Adult test, Your uploaded images) and "Sample" (50 images, 100 images, all images (slow)). Then four sliders, each with its own Reset button, and a Reset all.

| Slider                 | Range          | What it does                                     |
| ---------------------- | -------------- | ------------------------------------------------ |
| Brightness             | -0.50 to +0.50 | Adds or subtracts brightness.                    |
| Contrast               | 0.50 to 1.50   | 1.00 is unchanged. Below 1 flattens the picture. |
| Rotation               | -30° to +30° | Tilts the image.                                 |
| Crop (keep the center) | 60% to 100%    | Zooms in on the middle. 100% is unchanged.       |

After about a third of a second of not touching the sliders, the app re-runs the model on the sample, with a progress bar and "Recomputing 100 images... 45%". Moving a slider mid-run cancels the old run and starts a new one.

**Results.** A table with three rows:

| Row                          | Meaning                                                                                       |
| ---------------------------- | --------------------------------------------------------------------------------------------- |
| Our model, with your changes | Scores on the altered pictures.                                                               |
| Our model, original images   | Scores on the same sample, unaltered. This is your control.                                   |
| Always say "pneumonia"       | The lazy baseline for this sample (63.0% on the 100-image Kermany sample, not exactly 62.5%). |

Under the table: "8 of 100 predictions changed compared with the original images. Mean absolute change in P(pneumonia): 8.8%." with a short line explaining it, plus the compute time.

Example, brightness -0.40 on the Kermany sample of 100: accuracy 79.0% falls to 71.0%, sensitivity 100% to 98.4%, specificity 43.2% to 24.3%, balanced accuracy 71.6% to about 61%.

**Before and after.** A dropdown picks one image from the sample. Two big pictures, Original and "With your changes", each with the model's answer and confidence. Below is **Log the changed image**.

**What to demo.** Drag Brightness to -0.4. While it recomputes, show the Before/After pair: the same X-ray, the model "72% sure pneumonia" on the original and "95% sure pneumonia" on the dark one. The picture looks worse to a person, and the model got more confident, not less. That is the point.

**What to ask.** "What would a radiologist do with a dark image?" (Ask for a retake.) "What did the model do?" (Answered anyway.)

**Gotchas.**

- Perturb always runs MobileNet live, regardless of the precomputed switch. Expect about 10 to 15 seconds for 100 images on a laptop without a GPU. Choose "50 images" on slow machines. Avoid "all images (slow)" in class.
- The sample is fixed and stratified, so everyone sees the same images and the same pneumonia share.
- With "Your uploaded images" selected, the Sample dropdown is disabled and only labelled uploads get scores. If none are labelled the table is replaced by a hint.
- If there is no trained model the card says "No trained model yet. Train one in the Train tab first."

#### Stamp

This is the shortcut lab. It is a long panel with four numbered cards. A blue "Why this matters" note explains the idea.

**1. Design the stamp.**

- "Put the stamp on the training images of": **Pneumonia images** (default), Normal images, or Both classes. "Both classes" puts the marker on everything, so it carries no information. That is the fix.
- Size (8 to 96 px, default 48), Corner (top-left default), Offset x and Offset y (8 px), Opacity (100%).
- Two previews: "Without stamp" and "With stamp", showing a white "R" in the corner.
- **Use the default stamp** resets all of it. A grey line says "Default stamp + precomputed features: instant." when you are on the fast path, or "Custom stamp: the images are processed live on this laptop (slower, with progress)."
- A hint suggests a challenge for fast finishers: find the smallest stamp that still fools the model.

**2. Retrain on the stamped training set.** One button, **Stamp training set and retrain**. It trains a new model with the same recipe as the student's normal model (same epochs, seed, learning rate, balance, adult mix). Augmentation is not used here, and the panel says so. The student's normal model is kept as "Before". When done: "Stamp-trained model ready: stamp on pneumonia images, 48 px, top-left, 20 epochs." A **Discard the stamp-trained model** button is there if you want to start over.

**3. Flip rate.** Two dropdowns: "Test images" (Kermany test or Adult test) and "Stamp the TEST images of" (normal by default, pneumonia, both). The table updates by itself on the default stamp; otherwise press **Compute flip rate**.

Expected values on Kermany test, stamping the normal test images:

| Model                | Class        | Said "pneumonia" without stamp | with stamp | Flip rate          |
| -------------------- | ------------ | ------------------------------ | ---------- | ------------------ |
| Before: normal model | Normal (234) | 46.2%                          | 63.7%      | 17.5% (41 of 234)  |
| After: stamp-trained | Normal (234) | 1.3%                           | 94.9%      | 93.6% (219 of 234) |

Plain meaning of the flip rate: the share of images where the answer changed after the stamp was added. The stamp-trained model flips about 94% of healthy children to pneumonia just because a letter appeared in the corner. The normal model reacts a little too (17.5%), which is useful for comparison but not zero.

**The trap just below.** A second table shows accuracy on the clean, unstamped images:

| Predictor                  | Accuracy | Sensitivity | Specificity | Balanced |
| -------------------------- | -------- | ----------- | ----------- | -------- |
| Before: normal model       | 82.4%    | 99.5%       | 53.8%       | 76.7%    |
| After: stamp-trained model | 73.4%    | 58.2%       | 98.7%       | 78.5%    |

The shortcut model has a higher balanced accuracy than the honest one, so on clean data it looks as good or better. The tell is its sensitivity: it misses about 42% of unstamped pneumonia, because it learned "marker present means pneumonia" and not lung features. Without the stamp experiment, nobody would notice.

**4. Try it on single images.** A grid of 12 test thumbnails per page (24 in total) with "Show: normal test images / pneumonia test images". Click a picture to put the stamp on it, click again to remove it. A panel shows the picture and two answers: "Before (normal model)" and "After (stamp-trained model)". The best live demo: click a normal X-ray, stamp it, and watch the "After" box flip from "normal 100% sure" to "pneumonia". **Log this** saves the case and tags it "shortcut" when the stamp is on.

**What to demo.** The order on the screen is the order to teach: default stamp, retrain (3 seconds), read the flip table, read the clean-accuracy table, then do the single-image click.

**What to ask.**

- "Why did a model that is wrong this often still look good on the exam?" (Because the exam images had no stamps.)
- "Where could a real marker come from?" (A scanner label, a hospital tag, a portable machine's pattern.)
- "How would you fix this?" (Stamp both classes, or remove the marker.) Let them try "Both classes" and retrain.

**Gotchas.**

- **Keep the stamp at the default for the instant path.** Any change to size, corner, offsets or opacity makes the app extract features live for the stamped images. That means a progress bar and a wait of minutes on a slow laptop. Use custom stamps only for the "smallest stamp" challenge, on fast machines, and warn students.
- The same is true if the precomputed switch is off.
- Train the normal model first. The panel says so when there is none. The stamp-trained model does not replace it.
- If a student retrains the normal model after the stamp lab, the flip table shows "The models changed since this was measured. Press Compute flip rate again."
- The stamp-trained model is not part of the Results tab. The results card always says "Trained on stamped images: no". The stamp lab is a separate experiment.
- The stamp is a small white "R" drawn onto the image at 224 x 224 pixels. It is visible but small, and students find it hard to believe it matters. Say so.

#### Your images & webcam

**Your images.** A red notice: "Do not upload your own or family members' medical images. Images never leave this browser." A drop zone with **Choose images** (JPEG, PNG or WebP, up to 10 MB each, at most 50 images). Each picture is cropped to a centered square and resized to 224 x 224.

Each uploaded image gets a thumbnail, a label dropdown (**unlabeled**, **pneumonia**, **normal**), and **Remove**. **Clear all** removes everything.

"Predictions for your images" shows, per image, your label, the model's answer with a confidence bar, and **Log this**. If some images are labelled, a scores table (Our model and the baseline) appears with a note "Scores use only the N images you labeled." Below: "The model only knows two answers. Whatever you show it, it must pick pneumonia or normal (a closed world): it has no 'none of the above'."

Good uploads: a photo of a cup, a face, a landscape, a cartoon, a random meme. The closed-world point lands best with objects, not with X-rays of other body parts.

**Live webcam.** Buttons: **Start camera**, **Stop camera**, **Freeze & log**. A status text reads "Camera on" or "Camera off". Once on, the live feed fills a box with an overlay in the corner such as "pneumonia, 95% sure", updated about four times a second. A hint: "Try pointing it at a cup or your hand - the model must still pick a class."

**Freeze & log** captures the current frame, keeps it on screen with its prediction, and saves it straight to the failure log tagged "other", with an editor underneath to change the tag (to closed-world) and add a hypothesis. A counter reads "Frozen frames saved for the failure log: N".

**What to demo.** Hold up a cup. The model says "pneumonia 95% sure" or "normal 80% sure" and has no way to say "this is not an X-ray". Then slide the tau slider to 90% while the camera runs: some frames switch to orange "Refer to a doctor". That is a good moment to link to the human in the loop.

**What to ask.** "Did the model ever say 'I don't know'? Why can't it?"

**Gotchas.**

- The camera works only on `https://` or `localhost`. The hosted link is https. Opening the page from a file or a plain `http://` address on another machine shows "The camera needs a secure page".
- Camera denied: "Camera permission was denied. Click the camera or lock icon in the browser address bar, set Camera to Allow for this site, then reload the page and press Start camera again." Ask students to allow the camera once before the session.
- "The camera is busy" means another tab or app (a video call) is using it. "No camera found" means none was detected.
- Start camera is disabled until a model is trained.
- The camera stops by itself when you leave the panel, close the tab or reset. If a laptop's camera light stays on, press Stop camera.
- The webcam is not required for anything except the closed-world demo. Uploads cover the same point.

### 2.6 Failure log

**What students see.** A lead sentence, a "Counts by tag" card with seven tag chips plus "all", a "What do the tags mean?" fold, **Export JSON**, and a list of entries (newest first).

**The seven tags.**

| Tag          | Meaning in the app's words                                                                                | When to use it                                         |
| ------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| imbalance    | One class is much more common in the training data, so the model leans towards it.                        | Healthy children called pneumonia on the Kermany test. |
| shortcut     | The model reacts to a marker or artefact (a stamp, a label, a corner) instead of the lungs.               | Stamp flips.                                           |
| shift        | The image comes from a different population or machine than the training images (children vs adults).     | Adult test mistakes.                                   |
| quality      | The picture is darker, lower contrast, tilted or badly framed compared with the clean training images.    | Perturb and stress mistakes.                           |
| closed-world | The model must answer pneumonia or normal even when the picture shows something else (a cup, a hand).     | Webcam and non-X-ray uploads.                          |
| evaluation   | The score itself misleads: too few test images, the wrong metric, or test data that leaked into training. | The 16-image validation demo.                          |
| other        | None of the above, or not sure yet.                                                                       | Anything else.                                         |

Clicking a chip filters the list to that tag. Clicking it again clears the filter.

**An entry** shows a thumbnail of exactly what the model saw (including the stamp or the perturbation), a coloured tag, the source (for example "perturb (kermany_test)", "uploads", "stamp", "webcam"), a time, and a sentence such as "Truth: normal. Model said: pneumonia, 95% sure (P(pneumonia) = 95.0%). Wrong." If the case was referred it reads "Refer to a doctor (leaning pneumonia)". Below are the student's hypothesis and a grey line with the model settings in force ("realistic, 20 epochs, lr 1.0e-3, seed 42, refer below 50%, precomputed features"). Each entry has **Edit tag and hypothesis** and **Delete**.

**How entries get in.** There are four routes, and they behave a little differently:

| Where                                   | Button                | What happens                                                                                                   |
| --------------------------------------- | --------------------- | -------------------------------------------------------------------------------------------------------------- |
| Evaluate, prediction list; uploads list | Log this              | Opens a small form. The student must pick a tag and may type a hypothesis, then **Save to failure log**. |
| Break it, Perturb                       | Log the changed image | Same form. Saves the altered image.                                                                            |
| Break it, Stamp, single images          | Log this              | Saves at once. Tag is "shortcut" if the stamp is on, else "other". An editor appears to change it.             |
| Break it, webcam                        | Freeze & log          | Saves at once with tag "other". An editor appears.                                                             |

**Export JSON** downloads a file named like `vision-failures-20261101-1015.json`, containing every entry with its thumbnail and settings.

**What to demo.** After a few minutes of free breaking, project the log tab and read the tag counts aloud. On the board, write the seven tags and tally votes from students' own logs. That is your Day 1 debrief and Day 2 recap.

**What to ask.** "Which of your hypotheses could you actually test?" Push for one that names a change they could make and a number that would move.

**Gotchas.**

- The log holds at most 100 entries. When it is full, a yellow notice says to export and delete some.
- It is saved in the browser, so it survives a reload. Reset session clears it.
- If the browser refuses to store it (private mode, full storage), a notice warns that it only lives in this tab. Export before closing.
- Students often forget the hypothesis. Remind them: the tag says what, the hypothesis says why.

### 2.7 Results

**What students see.** A control card, then the "Results card" (this is what Print prints, and it mirrors the paper handout).

**Control card.**

- **Evaluate all three sets now:** scores any of Kermany test, Adult test and Kermany stress that have not been scored yet. It is greyed out when all three are done.
- **Re-run all three with the current settings:** redoes all three, for example after flipping the precomputed switch.
- **Download results JSON** (`vision-results-YYYYMMDD-HHMM.json`) and **Print**.
- A status line: "3 of 3 sets evaluated. Features: precomputed."

If the student has already visited each of the three sets on the Evaluate tab, they are already scored. The button is a shortcut for anyone who has not.

**Leaderboard value (worst-case score).** A big number in a blue box, with "Set that set the minimum", "Refer threshold (tau)" and "Coverage on that set". For the default model it reads **51.5%, Adult test, tau 50%, coverage 100%**. Until all three sets are scored it reads "Not complete yet. N of 3 test sets are evaluated."

What it is: the lowest of the three sets' half-credit scores. At tau 50% (nobody is referred), half-credit is the same as balanced accuracy, so it is the average of sensitivity and specificity, not plain accuracy. That is why the stress set contributes 60.7% even though its accuracy is 70.4%. A model is only as good as its worst set.

**Scores per test set.** A table with two rows per set (our model and the baseline) and columns Accuracy, Sensitivity, Specificity, Balanced accuracy, Referred n (%), Coverage, Half-credit score and Features (precomputed or live extraction). The set that set the minimum is tagged "(worst case)". The Features column matters if students compare numbers: the same model, same settings, can differ very slightly between precomputed and live runs.

**The model behind these numbers.** Balance mode, adult X-rays mixed in, augmentation, epochs, learning rate, seed, refer threshold, "Trained on stamped images: no".

**Failures you logged.** The same seven chips with counts.

**Snapshots.** A text box ("Before fix" suggested first, then "After fix 1", "After fix 2"), **Save snapshot**, and a counter "N of 8 saved". Save is disabled until all three sets are scored. Each snapshot becomes a column. Every column after the first shows the change in points against the previous column, in green or red. Snapshots keep the tau, mode and settings they were saved with.

Default model versus "balanced plus adult mix":

|                                                 | Before fix                | After fix 1                        |
| ----------------------------------------------- | ------------------------- | ---------------------------------- |
| Model                                           | realistic, adult mix off  | balanced, adult mix on             |
| Leaderboard value                               | 51.5% (Adult test)        | 59.5% (+8.0 pts, still Adult test) |
| Kermany test: acc / sens / spec / half-credit   | 82.4 / 99.5 / 53.8 / 76.7 | 83.8 / 97.7 / 60.7 / 79.2          |
| Adult test: acc / sens / spec / half-credit     | 51.5 / 90.0 / 13.0 / 51.5 | 59.5 / 41.0 / 78.0 / 59.5          |
| Kermany stress: acc / sens / spec / half-credit | 70.4 / 99.2 / 22.2 / 60.7 | 76.8 / 96.7 / 43.6 / 70.1          |

Notice what the fix traded away: sensitivity on the adult set fell from 90% to 41% while specificity rose from 13% to 78%. The score improved, but a sick adult is now missed more often than caught. A good point for the closer.

**What to demo.** Evaluate all three, point at the worst-case tag, then change one thing at the top of the card (the tau slider on Evaluate or Break it) and come back: the note "Recomputed for the current threshold (tau = 90%)" appears and the numbers update from stored predictions. Then show how a snapshot column works.

**What to ask.** "Which set is the weakest? What does that tell you to fix first?"

**Gotchas.**

- **Save the "Before fix" snapshot before changing the recipe.** Once a student toggles a fix and retrains, the old model and its evaluations are gone, and there is nothing left to snapshot. Sequence it: evaluate all three, save "Before fix", then fix, retrain, evaluate all three, save "After fix 1".
- Snapshots survive reload and are wiped by Reset session.
- If the precomputed switch was changed after some sets were scored, a yellow note says which sets used which mode and suggests re-running all three.
- Print hides the control buttons and the lead sentence, so the page prints as the card.
- Referral gives half credit, so raising the threshold rarely helps the leaderboard (see 2.4).

---

## 3. Glossary

Every term the UI uses, in plain words, with a number from this app where possible.

| Term                                               | Plain meaning                                                                                                                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backend**                                  | The part of the browser that does the maths: WebGL (graphics card, fastest), WASM (CPU, quick), CPU (slowest).                                                            |
| **P(pneumonia)**                             | The model's output: a number from 0 to 1. At 0.5 or above the model says pneumonia.                                                                                       |
| **Confidence**                               | How sure the model is of whatever it said: the larger of P and 1 minus P, so always 50% to 100%. A confident wrong answer is common.                                      |
| **Sensitivity**                              | Of the children who really have pneumonia, the share the model caught. Default model on the Kermany test: 388 of 390, 99.5%.                                              |
| **Specificity**                              | Of the healthy children, the share the model correctly cleared. Default: 126 of 234, 53.8%. On the adult test: 13%.                                                       |
| **Accuracy**                                 | Share of all cases called correctly. Misleading when classes are unequal: always answering pneumonia gives 62.5% on the Kermany test.                                     |
| **Balanced accuracy**                        | The average of sensitivity and specificity. Any model that ignores the picture scores 50%. Default on the Kermany test: 76.7%.                                            |
| **Confusion matrix**                         | A 2 by 2 table of right and wrong answers per true class. Four boxes: correctly cleared, false alarm, missed, caught.                                                     |
| **Always say "pneumonia" baseline**          | A pretend model that never looks at the picture. 62.5% on the Kermany test, 50% on the adult test. Your real model has to beat it.                                        |
| **Class weights**                            | How much each class counts in the training error. Realistic: 1.00 and 1.00. Balanced: normal 2.00, pneumonia 0.67, so a normal X-ray counts three times as much.          |
| **Realistic vs balanced**                    | The Data tab's class balance choice. Realistic keeps the natural 1:3 mix with equal weights. Balanced uses the class weights above.                                       |
| **Augmentation**                             | Adding randomly brightened, darkened, rotated or cropped copies of each training image (+2 views per image). It teaches the model that these changes are not the disease. |
| **Mix in adult X-rays**                      | A Day 2 fix: adds 100 pneumonia plus 100 no-finding adult images to the training set.                                                                                     |
| **Epoch**                                    | One pass over all training images. Default 20.                                                                                                                            |
| **Learning rate**                            | The size of each correction step. Default 0.001.                                                                                                                          |
| **Seed**                                     | The starting point for the random numbers. Same seed and settings give the same model, which is why the whole room can match. Default 42.                                 |
| **Loss**                                     | The error number the training is shrinking. Lower is better.                                                                                                              |
| **Training set**                             | The 1,020 images the model learns from.                                                                                                                                   |
| **Validation set**                           | Images used while training to watch progress. Ours has 180 images from the same pool, so it is optimistic: about 99%.                                                     |
| **Official validation set**                  | The dataset authors' 16-image set. One image is worth 6.25 points, so scores jump. Used only to show why a tiny validation set is untrustworthy.                          |
| **Test set**                                 | Images the model never saw. The Kermany test has 624. This is the honest score for the same kind of patient: about 82%.                                                   |
| **Adult test**                               | 200 adult X-rays from another hospital, 100 pneumonia and 100 "no finding". Default accuracy: 51.5%.                                                                      |
| **Stress set** (Kermany stress)              | The same 624 test images, made darker, lower in contrast, rotated 5° and cropped to 95%. Default accuracy 70.4%, balanced 60.7%.                                         |
| **Distribution shift**                       | The new data looks different from the training data (other age group, hospital or machine), and the score drops. Here from 82% to 51%.                                    |
| **Perturbation**                             | A deliberate change to the test picture: brightness, contrast, rotation or crop.                                                                                          |
| **Shortcut**                                 | A cue that predicts the label in the training set but is not the disease, such as a stamped "R". The stamp-trained model flips 93.6% of healthy test X-rays.              |
| **Stamp**                                    | The white "R" the app draws in a corner of selected images.                                                                                                               |
| **Flip rate**                                | Of the images that got the stamp, the share whose answer changed. 93.6% for the stamp-trained model, 17.5% for the normal one.                                            |
| **Closed world**                             | The model only knows two answers, so it must pick one for anything, a cup included.                                                                                       |
| **Refer to a doctor** (referral, abstaining) | The model declines to answer when its confidence is below the threshold chosen on the slider.                                                                             |
| **Tau**                                      | The name of that threshold. 50% means never refer.                                                                                                                        |
| **Referred**                                 | How many cases were handed to a doctor, and what percentage of the set.                                                                                                   |
| **Coverage**                                 | The share of cases the model answered. At tau 90% on the Kermany test: 74.8%.                                                                                             |
| **Half credit**                              | Scoring rule for the leaderboard: a referred case counts as half right. Referring everything gives exactly 50.                                                            |
| **Half-credit score**                        | Per set: the average over the two classes of (correct plus half the referred) divided by the class size. At tau 50% it equals balanced accuracy.                          |
| **Worst-case score**                         | The lowest half-credit score of the three sets. The leaderboard value. Default: 51.5% (adult test). After balanced plus adult mix: 59.5%.                                 |
| **Snapshot**                                 | A saved copy of all three sets' scores and the model settings, so before and after can sit side by side. Up to 8.                                                         |
| **Precomputed features**                     | MobileNet's 1,280 numbers per image, calculated once beforehand and shipped as files, so students skip the slow step. Same numbers as live extraction.                    |
| **Live extraction**                          | Running MobileNet on every image on the student's laptop. Slow but identical.                                                                                             |
| **MobileNet v2**                             | The frozen network that turns an image into 1,280 numbers.                                                                                                                |
| **Head**                                     | The small trainable layer on top of MobileNet. About 1,281 numbers.                                                                                                       |
| **Failure log / tag / hypothesis**           | The record of a failure, the category of why it failed, and the student's testable guess about why.                                                                       |

---

## 4. Run of show

Times follow the workshop plan. Each block says where to be in the app and what numbers to expect. Anything marked "no app" is a slide-only block.

### Before anyone arrives

- Open the live link on the projector laptop in Chrome. Check the header: WebGL and Ready.
- Press Train, then open Evaluate on all three sets to confirm 82.4%, 51.5% and 70.4%. Then open Settings and press **Reset session** so the room starts clean.
- Allow the camera once, so the permission prompt is not a surprise.
- Have the USB copy and `?backend=wasm` ready.
- Confirm the adult test set loads (Data tab, Adult test row shows 100 and 100).
- Say the disclaimer aloud.

### Day 1: Train it, then doubt it (1h50)

| Time         | Block                         | Where in the app                             | What to click and say                                                                                                                                                                                                                                                                                                  | Numbers to expect                                                                                                                                                                |
| ------------ | ----------------------------- | -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0:00 to 0:10 | Hook                          | No app                                       | Vote on three X-rays, show the chatbot screenshots, reveal true labels.                                                                                                                                                                                                                                                |                                                                                                                                                                                  |
| 0:10 to 0:30 | Foundations                   | No app (or the Data gallery for one picture) | Concepts one to six. One quick look at the gallery to show "an image is numbers".                                                                                                                                                                                                                                      |                                                                                                                                                                                  |
| 0:30 to 0:40 | Setup                         | **Data**                               | Students open the link and wait for Ready. Read the counts table together. Open Workshop steps. Check camera permission on the Break it tab if time.                                                                                                                                                                   | 255 / 765 train, 45 / 135 validation, 234 / 390 test, 3 pneumonia per 1 normal.                                                                                                  |
| 0:40 to 1:05 | Train v1                      | **Train**, then **Evaluate**     | Everyone presses Train. Talk over the loss curve about transfer learning. Then Evaluate, Kermany test. Read the baseline row first, then accuracy, then sensitivity and specificity. Show the confusion matrix. Switch to Adult test as a teaser. Students write accuracy, sensitivity and specificity on the handout. | Training about 3 s, validation 98.9%. Kermany: 82.4% accuracy, 99.5% sensitivity, 53.8% specificity, 108 false alarms, 2 missed. Baseline 62.5%. Adult: 51.5%.                   |
| 1:05 to 1:15 | Break                         |                                              | Leave the slides up.                                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                  |
| 1:15 to 1:40 | Predict, then break (round 1) | **Break it**                           | Hand out prediction cards. Demo one Perturb slider (brightness -0.4) and the webcam on a cup once. Students then work freely: Perturb, Your images & webcam, and the Log this buttons on Evaluate. Optional for fast groups: Train, Repeat x5, to show the 16-image swing.                                             | Brightness -0.4: about 8 of 100 answers change, accuracy about 79% to 71%. Cup: a confident class. Repeat x5: 16-image spread about 6 points, 180-image about 0.6, test about 2. |
| 1:40 to 1:50 | Debrief                       | **Failure log**                        | Project the log, read the tag counts, sort failures on the board. Teaser: "A model once learned which hospital, not which disease."                                                                                                                                                                                    | Whatever the room found. Typical tags: imbalance, quality, closed-world.                                                                                                         |

Day 1 notes:

- If you are tight, drop the Repeat x5 demo. Keep the baseline comparison.
- The Workshop steps checklist lists "Evaluate on the adult X-rays" on Day 1. That is the quick teaser at the end of Train v1. The full shift lab is on Day 2.
- Students often keep their models between days if they use the same laptop and browser. That is fine; you do not need to reset.

### Day 2: Shortcuts, shift and the human in the loop (about 1h50)

| Time         | Block           | Where in the app                                        | What to click and say                                                                                                                                                                                                                                                                                                       | Numbers to expect                                                                                                                                                                             |
| ------------ | --------------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0:00 to 0:10 | Recap and story | **Failure log**                                   | Show yesterday's tag counts. Tell the Zech 2018 hospital story. If a laptop lost its model (a new browser), press Train on defaults, seed 42; it reproduces yesterday's numbers.                                                                                                                                            |                                                                                                                                                                                               |
| 0:10 to 0:40 | Shortcut lab    | **Break it**, **Stamp**                     | Leave the stamp at the default. Press **Stamp training set and retrain**. Read the flip table. Read the clean-accuracy table. Then click a normal X-ray in step 4 and watch "After" flip. Students record the flip rate. Challenge for fast finishers: smaller stamp. Try "Both classes" and retrain to show the fix. | Flip rate 93.6% (219 of 234) for the stamp-trained model, 17.5% for the normal one. Clean test for the stamp model: accuracy 73.4%, sensitivity 58.2%, specificity 98.7%.                     |
| 0:40 to 1:00 | New population  | **Evaluate** (Adult test), then **Results** | Open Evaluate on Adult test. Record the drop. Ask for two differences between the datasets. Then go to Results, press **Evaluate all three sets now** and save a snapshot named "Before fix".                                                                                                                         | Adult 51.5% (sensitivity 90%, specificity 13%, 87 of 100 no-finding patients called pneumonia). Results card: 51.5%, Adult test.                                                              |
| 1:00 to 1:10 | Break           |                                                         |                                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                                               |
| 1:10 to 1:35 | Fix it          | **Data**, **Train**, **Results**      | In Data, tick at least two fixes: Balanced, Mix in adult X-rays, Augmentation. Train again. Go to Results, press **Evaluate all three sets now**, **Save snapshot** ("After fix 1"). Try the refer slider and note it barely changes the worst-case value. Post the worst-case score to the leaderboard.        | Balanced plus adult mix: 59.5% worst case, adult test still the weakest (+8.0 points). Stress half-credit 60.7% to 70.1%. Kermany test 83.8%. Adding augmentation should lift stress further. |
| 1:35 to 1:45 | Trend           | No app                                                  | Big chatbots are not immune. Cut this first if running late.                                                                                                                                                                                                                                                                |                                                                                                                                                                                               |
| 1:45 to 1:55 | Closer          | No app (maybe Results on screen)                        | AI proposes, the doctor decides, with an audit trail. Post-quiz, feedback.                                                                                                                                                                                                                                                  |                                                                                                                                                                                               |

Day 2 notes:

- Do not skip the "Before fix" snapshot. Without it there is no before/after column.
- Remember to tell students to retrain with the same seed so the comparison is fair.
- If you ran Reset session on Day 2 morning, the Day 1 failure log is gone. Do not reset until after the recap.
- If Day 1 overran, cut the Trend block first; never cut the stamp lab or the re-measure step.

---

## 5. If something goes wrong

### Symptoms and fixes

| Symptom                                                                    | Likely cause                                                                         | Fix                                                                                                                                                                                                                                           |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Page is blank or never leaves "Loading"                                    | Outdated cache, blocked files, or a browser too old                                  | Hard reload (Ctrl+Shift+R). Try Chrome or Edge. If it persists, use the USB copy, served through `localhost` (not by double-clicking `index.html`).                                                                                       |
| Header shows `No backend` or a red "No compute backend available" banner | The browser cannot start any compute backend                                         | Try another browser. Add `?backend=wasm` to the address.                                                                                                                                                                                    |
| Header badge is yellowish and says WASM or CPU                             | The laptop has no usable WebGL                                                       | Fine. Training is still seconds with precomputed features. Perturb and live extraction are slower. Use the 50-image sample.                                                                                                                   |
| "WebGL context lost" banner                                                | The GPU driver reset, often with many tabs open                                      | Reload, or open with `?backend=wasm`.                                                                                                                                                                                                       |
| Camera does not open                                                       | The page is not https or localhost, or permission denied, or the camera is in use    | Use the hosted https link. Click the lock icon in the address bar, set Camera to Allow, reload. Close video calls. The webcam is only needed for one demo; uploads do the same job.                                                           |
| Training is slow (minutes, "Running MobileNet on image 200 of 1200")       | Precomputed switch is off or the files failed to load                                | Settings, switch "Use precomputed features" on. On Data, the green dot should say "Using precomputed features". If you see "Precomputed features could not be loaded", press **Try again**.                                             |
| Training stretches out for no reason                                       | The student is on a different browser tab, which throttles the page                  | Ask them to stay on the app tab.                                                                                                                                                                                                              |
| The Stamp tab is slow after the student changed the stamp                  | A custom stamp forces live extraction                                                | Press "Use the default stamp". Keep the precomputed switch on.                                                                                                                                                                                |
| Perturb slider takes 10 to 15 seconds                                      | Perturb always runs MobileNet live                                                   | Choose "50 images". Pause between slider moves.                                                                                                                                                                                               |
| Tab is sluggish or memory-heavy after about an hour                        | Accumulated state in a long session                                                  | Settings, **Reset session**. Export the failure log first if it matters.                                                                                                                                                                |
| A student is stuck mid-run                                                 | A cancelled or failed job left the page odd                                          | Reload the page. The trained model and the log come back. If still odd, **Reset session**.                                                                                                                                              |
| Student's numbers differ from yours                                        | Different seed, recipe, precomputed switch, tau, or a model restored from last visit | Compare the grey "Recipe" line on Train and the "The model behind these numbers" block on Results. Reset seed to 42, set tau to 50%, switch precomputed on, press Train again. Live extraction can differ from precomputed by a small amount. |
| Results shows "Not complete yet"                                           | One or more of the three sets has not been scored                                    | Press **Evaluate all three sets now**.                                                                                                                                                                                                  |
| Results numbers are gone after retraining                                  | Training again discards every stored evaluation                                      | Evaluate all three again.                                                                                                                                                                                                                     |
| The stamp flip table says "The models changed since this was measured"     | The normal model was retrained after the stamp model                                 | Press **Compute flip rate**, or retrain the stamp model.                                                                                                                                                                                |
| Failure log full (100 entries) or a storage warning                        | The log is capped, or the browser blocks storage                                     | Export JSON, delete entries. For storage warnings, export before closing the tab.                                                                                                                                                             |
| Internet drops mid-session                                                 | Wi-Fi                                                                                | Nothing to do once the page has loaded. A reload without internet fails, so do not reload. If a student must, give them the USB copy.                                                                                                         |
| Everything breaks                                                          |                                                                                      | Run the session from your own laptop on the projector and have students watch.                                                                                                                                                                |

### URL parameters

Add them to the end of the web address.

| Parameter          | Effect                                                                               |
| ------------------ | ------------------------------------------------------------------------------------ |
| `?backend=wasm`  | Use WebAssembly instead of WebGL. The header tooltip says "Forced by ?backend=wasm". |
| `?backend=cpu`   | Use plain CPU. Slowest, last resort.                                                 |
| `?backend=webgl` | Force WebGL first. Falls back automatically if it cannot start.                      |
| `?debug=1`       | Shows a small green box in the bottom right: "tensors: 106                           |

Combine them with `&`, for example `?backend=wasm&debug=1`.

### Running from the USB copy

Build once ahead of time (`npm run build` in `app/`), copy `app/dist` to at least two sticks, and from any laptop run `node dist/serve-offline.mjs`, then open `http://127.0.0.1:8080`. `npx serve dist` and `python -m http.server` also work. The camera works on `localhost`.

---

## 6. Cheat sheet: the numbers to remember

Default run: seed 42, 20 epochs, realistic training, precomputed, tau 50%.

| #  | Number                                                                                         | Where it appears                                    | What it proves                                                                                                                                                     |
| -- | ---------------------------------------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1  | **62.5%**                                                                                | Evaluate, baseline row on Kermany test (390 of 624) | A model that never looks at the picture already scores this. Every score is judged against it.                                                                     |
| 2  | **1:3** (255 normal, 765 pneumonia)                                                      | Data tab                                            | The training data is imbalanced, so the model leans toward answering pneumonia.                                                                                    |
| 3  | **99% validation** (98.9%)                                                               | Train tab, status line and charts                   | The in-sample check looks perfect. It is optimistic because it comes from the same pool as training.                                                               |
| 4  | **82% test** (82.4%), sensitivity 99.5%, specificity 53.8%                               | Evaluate, Kermany test                              | The honest score for the same kind of patient. Almost all the damage is healthy children called pneumonia (108 of 234).                                            |
| 5  | **51% adult** (51.5%, specificity 13%)                                                   | Evaluate, Adult test                                | Distribution shift. Barely better than the 50% baseline. 87 of 100 no-finding adults flagged.                                                                      |
| 6  | **61% stress balanced** (60.7%; accuracy 70.4%)                                          | Evaluate, Kermany stress                            | Darkening, flattening and tilting the picture slightly already breaks the model. Accuracy flatters it.                                                             |
| 7  | **94% stamp flip** (93.6%, 219 of 234)                                                   | Break it, Stamp                                     | The model learned the marker, not the disease. The normal model flips only 17.5%.                                                                                  |
| 8  | **42% of unstamped pneumonia missed** (sensitivity 58.2%)                                | Break it, Stamp, clean-accuracy table               | The shortcut model fails on clean data too. Its balanced accuracy (78.5%) still looks fine, which is why shortcuts hide.                                           |
| 9  | **6-point val16 spread** (6.25 points per image) against about 0.6 for the 180-image set | Train, Repeat x5                                    | A 16-image validation set is noise. Never choose a model with it.                                                                                                  |
| 10 | **Worst case 51.5, then 59.5**                                                           | Results                                             | The leaderboard value is the weakest of the three sets (adult test). Balanced plus adult mix lifts it by about 8 points but still leaves adult sensitivity at 41%. |

Two more worth having in your pocket:

- **Refer everything scores exactly 50.** Half credit means abstaining cannot game the leaderboard.
- **At tau 90% on the Kermany test:** accuracy 90.4%, 157 cases (25.2%) referred, coverage 74.8%, half-credit score 74.6%. The model looks better and scores worse.

---

## Appendix: UI behaviours that may surprise you

These are quirks, not bugs. Know them before you are in front of the room.

1. **Training pool size.** The facilitator notes say 1,200 training images. The Data tab shows 1,020 for training and 180 for validation, which together make 1,200.
2. **Checklist versus agenda.** The Workshop steps checklist puts "Evaluate on the adult X-rays" on Day 1 and "Test on the adult set and note the drop" on Day 2. The agenda treats the shift lab as Day 2. Present Day 1's adult look as a teaser.
3. **Accuracy rises when you refer.** With the slider above 50%, Accuracy, Sensitivity and Specificity use only answered cases, so they go up. The leaderboard half-credit score does not. Students may think referral "improves" the model. Show both tabs side by side.
4. **Leaderboard value is balanced accuracy, not accuracy.** It is the average of sensitivity and specificity at tau 50%. On the stress set the table says 70.4% accuracy and 60.7% half-credit.
5. **Shortcut model looks better on one metric.** On clean test images the stamp-trained model has a higher balanced accuracy (78.5%) than the honest model (76.7%), even though its sensitivity collapses to 58.2%. Point at sensitivity.
6. **The normal model also reacts to the stamp.** It flips 17.5% of normal test images. It is far less than 93.6%, but not zero.
7. **The stamp-trained model is not on the Results tab.** The results card only ever scores the main model and always prints "Trained on stamped images: no".
8. **Retraining wipes evaluations.** Any new head clears every stored evaluation, so Results must be re-run. The Data tab's recipe edits do not affect an existing model until Train again is pressed.
9. **Saved state outlives a reload.** The trained model, tau, settings, steps, failure log and snapshots persist in the browser. A laptop used for rehearsal opens with a restored model ("A trained model from your last visit was restored"). Reset session is the only way to clear all of it, and it clears everything at once.
10. **Perturb ignores the precomputed switch.** It always runs MobileNet live, so it is the slowest interactive part even on a good setup. Its baseline is 63.0% on the 100-image sample, not 62.5%, because it is a sample.
11. **Repeat x5 ignores the seed box.** It always uses seeds 1 to 5, and leaves the current model untouched.
12. **Log this behaves differently by location.** Evaluate and Perturb ask for a tag before saving. Stamp and webcam save immediately with a default tag ("shortcut" or "other"), and show an editor afterwards. If students forget to edit, the log fills with "other".
13. **"normal" versus "no finding".** Adult negatives are "no finding" everywhere. Uploads use the labels "pneumonia" and "normal" regardless of what the picture is.
14. **Natural letter markers.** Many X-rays already carry small "R" or "L" letters in a corner. The stamp is a much larger, white, deliberate "R".
