# Guide to "Build a Vision Model, Then Break It"

This guide explains every tab, button and number in the app, suggests two sessions of exercises, and lists what to do when something goes wrong. You do not need to know much machine learning. Every technical term is explained the first time it appears, and the glossary near the end collects them all.

**Educational only. Not for clinical use.** The app says this in the header, the footer and on the results card, and it means it. The model is a teaching toy trained on a small research dataset.

Contents:

1. [The app in one minute](#1-the-app-in-one-minute)
2. [Tab by tab](#2-tab-by-tab)
3. [Glossary](#3-glossary)
4. [Suggested exercises](#4-suggested-exercises)
5. [How to read a failure and write a hypothesis](#5-how-to-read-a-failure-and-write-a-hypothesis)
6. [Troubleshooting, URL parameters and data sources](#6-troubleshooting-url-parameters-and-data-sources)
7. [For instructors](#7-for-instructors)

---

## 1. The app in one minute

The app trains a pneumonia detector on children's chest X-rays, inside your browser, in a few seconds. Then it gives you tools to find out where the detector fails and why. Getting a high score is easy. Finding out whether the score means anything is the point.

### What is frozen and what is trained

| Part | Frozen or trained? | What it does |
| --- | --- | --- |
| MobileNet v2 | Frozen. It never changes. | A small image network that Google trained on everyday photos. It turns each X-ray into 1,280 numbers that describe the picture. |
| The head | Trained by you, in a few seconds | One layer (a logistic regression) that turns the 1,280 numbers into P(pneumonia), a probability between 0 and 1. At 0.5 or above, the model says "pneumonia". |

So pressing Train only fits the head, which has about 1,281 adjustable numbers. That is why it takes a few seconds and why you can retrain as often as you like. Using a big network that someone else trained, and teaching only a small last layer, is called transfer learning.

### What stays on your computer

Nothing leaves the browser. The X-rays, the MobileNet model and the precomputed features are files that come with the page, and once the page has loaded the app makes no further network requests. The footer says: "Runs entirely in your browser. X-rays and models never leave this computer."

Your own work is stored in the browser on that computer: the trained model, the failure log, snapshots, settings and ticks on the checklist. It survives a page reload. The "Reset session" button in the settings drawer wipes it. Uploaded pictures and webcam frames are kept in memory only and disappear when you reload.

### The layout

- **Header:** the title, a backend badge, a model badge and a Settings button. The backend badge shows `WebGL`, `WASM`, `CPU`, `No backend` or `Detecting...`; it is the part of the browser doing the maths (see the glossary). The model badge shows `Loading model...` and then `Ready`. You cannot train until it says Ready.
- **Six tabs:** Data, Train, Evaluate, Break it, Failure log, Results. Work flows left to right, but you can jump around.
- **Workshop steps** (a collapsed bar under the tabs) and the **Settings** drawer, both described below.
- **Footer:** the disclaimer and the privacy line.

### Two notices you may see

- A yellow bar with a Dismiss button, usually: "Precomputed features could not be used, so features are being extracted live on this laptop (slower, same result)." Training still works, only slower.
- A red banner such as "No compute backend available", "Model not available" or "WebGL context lost". The troubleshooting table has the fixes.

---

## 2. Tab by tab

All numbers in this guide come from the default run: seed 42, 20 epochs, realistic 1:3 training, precomputed features on, refer threshold at 50%. Your numbers may differ by a point or two with a different seed or browser backend (see the note under [Suggested exercises](#4-suggested-exercises)).

### 2.1 Data

**What you see.** A sentence about the 1:3 ratio (three pneumonia X-rays for every normal one), a status dot, a table called "How many images?", a "How should we train?" card and a gallery.

The status dot reads "Using precomputed features" when everything is fine. It reads "Live extraction ..." when precomputed features are switched off or could not be loaded.

**The count table.**

| Set | Normal | Pneumonia | Total | What it is for |
| --- | --- | --- | --- | --- |
| Kermany train | 255 | 765 | 1020 | Training images (children) |
| Kermany validation (ours) | 45 | 135 | 180 | Checking progress while training |
| Kermany validation (official) | 8 | 8 | 16 | Only used in the "dishonest evaluation" demo |
| Kermany test | 234 | 390 | 624 | The final exam, never used for training |
| Adult train pool | 100 (no finding) | 100 | 200 | Optional: mix into training |
| Adult test | 100 (no finding) | 100 | 200 | Different hospital, adults (the shift test) |

A line under the table points out that "Kermany stress" is not extra data. It is the same 624 test X-rays, made darker, lower in contrast, rotated 5 degrees and cropped to 95%, always in exactly the same way so everyone gets the same stress.

Adult images that are not pneumonia are called **no finding** everywhere in the app, not "normal". They are adults whose radiology report mentioned none of the 14 listed conditions, which is a weaker claim than "healthy". The labels were mined from reports by software, so a few are wrong.

**The "How should we train?" card.**

- **Class balance.** *Realistic* counts every image the same, so the common class (pneumonia) dominates. *Balanced* makes mistakes on the rarer class count more. The line "Class weights used in this mode" shows normal 1.00 / pneumonia 1.00 for Realistic. Balanced would use normal 2.00 / pneumonia 0.67, which means one normal X-ray counts three times as much as one pneumonia X-ray when the training error is added up. That cancels the 1:3 imbalance.
- **Day-2 fixes.** Two checkboxes. "Mix in adult X-rays (+200)" adds 100 pneumonia and 100 no-finding adult images to the training set. "Augmentation (+2 views per image)" adds two randomly brightened, rotated or cropped copies of every training image. Leave both off for your first model.
- **Training will use N rows.** 1020 rows by default (255 normal, 765 pneumonia). With the adult mix it is 1,220. With augmentation each image counts three times.

**The gallery.** "Browse the X-rays" has two dropdowns (Set and Label) and Previous/Next buttons. Each thumbnail carries a green "normal" or red "pneumonia" chip. Many children's X-rays carry a small "R" or "L" side marker in a corner. It appears on both classes, so it carries no information, which makes it a useful contrast with the deliberate stamp in the Break it tab.

**What to try.**

1. Filter to Kermany train, normal, then pneumonia. Can you tell them apart by eye? Most people cannot.
2. Read the 255 and 765. Say "three to one" before you see any score.
3. Filter to Adult test and flip through a page. What looks different? (Bigger bodies, different contrast, a different kind of machine.)
4. Work out: if a model always said "pneumonia", what accuracy would it get on the test set? 390 out of 624, which is 62.5%.

**Common mistake.** Changing the recipe here does not change a model that is already trained. See "Model in use" below.

### 2.2 Train

**What you see.** An intro sentence, a recipe card, two charts, an "Advanced settings" fold and a "Repeat x5 with new seeds" card.

**The recipe card** repeats the recipe chosen on the Data tab. Under "Check progress on" you pick which validation set the progress line is drawn on: "Our validation set (180 images)" (default) or "Official validation set (16 images)". This never touches the test set.

Press **Train**. It becomes **Train again** after the first run. The status line reads "Training: epoch 7 of 20, 1.2 s elapsed" and then "Finished 20 epochs in 3.0 s. Validation accuracy 98.9%." **Cancel** stops a run and keeps your previous model.

**The charts.**

| Chart | How to read it |
| --- | --- |
| Loss (lower is better) | The error the training is trying to shrink. Solid blue is the training images, dashed orange is the validation images. Both fall from about 0.27 to about 0.05 and flatten. |
| Accuracy (higher is better) | The share answered correctly. Both lines climb to about 99%. |

Both lines finishing near 99% is the trap. It looks like a superb model. But the validation set comes from the same pool of children as the training set, so it flatters the model. Remember that 99% because the next tab takes it away.

**Advanced settings.**

- **Epochs** (passes over the training data), 20 by default.
- **Learning rate** (the size of each correction step), 1.0e-3 by default.
- **Seed** (the starting point for the random numbers), 42 by default, with a **New seed** button. The same seed and the same settings give the same model.

**Repeat x5 with new seeds.** Trains the same recipe five times with seeds 1 to 5 and shows how much the score moves:

| | Validation accuracy, 16 images | Validation accuracy, 180 images | Kermany test accuracy |
| --- | --- | --- | --- |
| Lowest to highest | about 81% to 87% | about 99% | about 79% to 81% |
| Spread | about 6 points | under 1 point | about 2 points |

One image on a 16-image set is worth 6.25 points, so its score jumps around and the jumps are not information. If you picked your "best" model using that set, you would be choosing noise. This is the "dishonest evaluation" demo. It never replaces your model, ignores the seed box, and throws its five models away.

**Common mistakes.**

- Treating the 99% as the real score. It is the validation score. The honest one is on the Evaluate tab.
- Switching to another browser tab while training. Browsers slow down background tabs, so a three-second run can stretch out.
- Expecting a fresh start. If you used the app before, a model from your last visit is restored, and the status line says "A trained model from your last visit was restored. Train again to replace it."

### 2.3 The "Model in use" strip

At the top of Train, Evaluate, Break it and Results there is a small grey strip that tells you which model the numbers on screen belong to. For example:

> Model in use: Realistic classes, adult X-rays off, augmentation off, seed 42, 20 epochs, learning rate 1.0e-3. Trained 09:01 AM (took 4.5 s). Validation accuracy 98.9% on our 180-image validation set.

If the model came from an earlier visit, the strip adds "Restored from an earlier session." If there is no model, the other tabs show "No model yet: train one on the Train tab."

The strip also protects you from a very common mistake. If you change the recipe on the Data tab and forget to press Train again, an amber line appears: "The recipe on the Data tab has changed since this model was trained. Press Train again to apply it." with a **Go to Train** button. Until you retrain, every score describes the old recipe.

When your numbers differ from someone else's, compare this strip first. Seed, recipe, epochs and the precomputed switch are the usual causes.

### 2.4 Evaluate

**What you see.** "Test on" with four options: **Kermany test**, **Adult test**, **Kermany stress** and **Your uploaded images**. Under it come a one-line description, the "Refer to a doctor" slider and the results.

There is no Evaluate button. Scoring runs by itself when you open the tab with a trained model, and again each time you pick a different set. With precomputed features it is nearly instant. Without a model, the tab says "No trained model yet" and offers a Go to Train button.

**The metrics table** for the Kermany test with the default model:

| Predictor | Accuracy | Sensitivity | Specificity | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Our model | 82.4% | 99.5% | 53.8% | 76.7% |
| Always say "pneumonia" | 62.5% | 100.0% | 0.0% | 50.0% |

- **Accuracy** is the share of all 624 X-rays called correctly. Compare it with the baseline row, never with 100%.
- **Sensitivity** is the share of sick children the model caught. Here 388 of 390.
- **Specificity** is the share of healthy children the model correctly cleared. Here 126 of 234. The other 108 were false alarms.
- **Balanced accuracy** is the average of sensitivity and specificity. Any model that ignores the picture scores exactly 50% on it, which makes it fairer than accuracy when one class is more common.
- The **"Always say pneumonia"** row is a pretend model that never looks at the picture. Your model has to beat it to be worth anything.

A line under the table says whether this evaluation used precomputed features or features "extracted live on this laptop".

**The confusion matrix.** Rows are the true class, columns are what the model said. Green cells are right, red cells are wrong. Each cell shows a count, a percentage of its row and a plain word.

| | Model said normal | Model said pneumonia |
| --- | --- | --- |
| True normal (234) | 126 correctly cleared | 108 false alarm |
| True pneumonia (390) | 2 missed | 388 caught |

A note below explains that a missed sick child usually costs more than a false alarm, because a false alarm means an extra check while a miss can mean a child goes home untreated.

**The three test sets with the default model:**

| Set | What it tests | Accuracy | Sensitivity | Specificity | Balanced | Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| Kermany test (624) | Patients like the training ones | 82.4% | 99.5% | 53.8% | 76.7% | 62.5% |
| Adult test (200) | A new population (shift) | 51.5% | 90.0% | 13.0% | 51.5% | 50.0% |
| Kermany stress (624) | The same X-rays, degraded | 70.4% | 99.2% | 22.2% | 60.7% | 62.5% |

The adult baseline is 50% because that set was hand-balanced at 100 and 100. In real NIH data pneumonia is about 1.2% of images. On the adult set the labels change to "sick patients caught" and "patients with no finding cleared".

**Individual predictions.** Under the matrix is a paged list of every X-ray in the set.

- A line above the list gives the totals, for example "Mistakes 110 | Referred 0 | Correct 514 of 624 X-rays".
- **Show** chips choose the view: *Mistakes first (most confident first)* (the default), *Only mistakes*, *Only referred* (it appears only when the threshold is above 50%), *Only correct* and *Dataset order*. Each chip shows its count.
- **True label** chips filter to *All*, *pneumonia* or the negative class (*normal* or *no finding*). The counts on the Show chips follow this filter.
- A pager has First, Previous, Next and Last buttons and "Rows per page" of 20, 40 or 100.
- Each row shows a thumbnail, the image id, the true label, "Model says" (with "(wrong)" when it is wrong, or "Refer to a doctor"), the confidence, a **Log this** button and a **Heatmap** button.

"Most confident first" matters because a confident wrong answer is more instructive than a borderline one. **Confidence** is how sure the model is of the class it chose, always between 50% and 100%.

On the Kermany stress set the small thumbnail in each row is the original picture, while Log this and Heatmap use the degraded picture the model actually saw.

**Heatmaps.** Pressing **Heatmap** on a row opens a Grad-CAM view. Grad-CAM colours the parts of the image that pushed the model towards an answer; warm colours mean more evidence. Some details:

- It opens on the class the model **predicted** ("Evidence for pneumonia" for a pneumonia call, "Evidence for normal" for a normal call). You can switch with the "Evidence for" buttons.
- "Heatmap strength" fades the colours in and out. "Hide heatmap" closes it.
- The map is only 7 by 7 cells, so it is blurry. It is a hint about where to look, not proof of how the model reasons. One heatmap is an anecdote.
- Heatmaps are also available on single images in the Stamp lab and for your own uploads.

**Your uploaded images.** Choosing this swaps the results for the upload manager, described in section 2.5.

**What to try.**

1. Read the baseline row first, then your model's row, then specificity.
2. Find the 108. Which direction do almost all the mistakes go? (Healthy children called sick.)
3. Switch to Adult test, then Kermany stress. Which number drops the most, and which one hides the damage?
4. Filter to *normal* and *Only mistakes*, then open three heatmaps. Compare them with three correctly cleared normal children.

**Common mistakes.**

- Evaluating before training. The tab only shows the empty message.
- Training again discards every saved evaluation. Evaluate and Results recompute for the new model.
- Quoting 99% (validation) as if it were the test score.

### 2.5 The "Refer to a doctor" slider

This slider appears on Evaluate and on Break it. It is one shared setting, so moving it in one place moves it in the other. It is remembered across reloads.

The model always has a lean: P(pneumonia) at or above 50% means "pneumonia", below means "normal". Its confidence is the larger of P and 1 minus P. The slider sets a bar for confidence. When the confidence is below the bar, the app does not answer. It says "Refer to a doctor" and shows the case in orange.

- At 50%, the leftmost position and the default, the label reads "50% = the model always answers (never refers)".
- The slider goes up to 99%.
- Raising it does not make the model more accurate. It makes the model say less, and the cases it keeps are the easier ones.

Two columns appear once the slider is above 50%. **Referred** is how many cases went to a doctor and what share of the set that is. **Coverage** is the share of cases the model dared to answer. Accuracy, sensitivity and specificity are then computed only on the answered cases.

On the Kermany test at 90%, accuracy on answered cases rises to about 90%, 157 cases (25.2%) are referred and coverage is 74.8%. The matrix title reads "467 X-rays (157 referred, not counted)".

**Half credit.** Accuracy on answered cases always looks better when the bar rises, because the model picks its easy cases. To stop abstaining from being a trick, the worst-case score on the Results tab gives half a point for every referred case. A model that refers everything scores exactly 50, the same as always saying "pneumonia". On the same 90% setting the Kermany half-credit score is about 74.6%, lower than the 76.7% at 50%. Accuracy went up and the score went down.

So the threshold buys safety, not score. On the adult set the model is so often wrong with high confidence that raising the bar changes the score very little. Abstaining cannot rescue a model that is confidently wrong.

### 2.6 Break it

The tab has a short lead, three sub-buttons (**Perturb**, **Stamp**, **Your images & webcam**), the shared slider and then the chosen experiment. All three need a trained model. Break it is a way of working as much as a tab: imbalance and dishonest evaluation are found on Data, Evaluate and Train, while the other four breaks live here.

#### Perturb (image quality)

**What you see.** "Test images" (Kermany test, Adult test or Your uploaded images), "Sample" (50 images, 100 images or all images (slow)), then four sliders, each with its own Reset button, and Reset all.

| Slider | Range | What it does |
| --- | --- | --- |
| Brightness | -0.50 to +0.50 | Makes the picture darker or lighter. |
| Contrast | 0.50 to 1.50 | 1.00 is unchanged. Below 1 flattens the picture. |
| Rotation | -30 to +30 degrees | Tilts the image. |
| Crop (keep the center) | 60% to 100% | Zooms into the middle. 100% is unchanged. |

A third of a second after you stop moving a slider, the app runs the model again on the sample and shows a progress bar ("Recomputing 100 images... 45%"). Moving a slider during a run cancels the old run and starts a new one.

**The table** has three rows. *Our model, with your changes* is the score on the altered pictures. *Our model, original images* is the same sample unaltered, which is your control. *Always say "pneumonia"* is the baseline for this sample (63.0% on the 100-image Kermany sample, not exactly 62.5%, because it is a sample). Below it: "8 of 100 predictions changed compared with the original images" and the mean change in P(pneumonia).

With brightness at -0.40 on the 100-image Kermany sample, about 8 answers change, accuracy falls from about 79% to about 71%, and balanced accuracy falls from about 72% to about 61%. Rotation of 20 degrees takes balanced accuracy to about 53%, and a 70% crop to about 50%, which is a coin flip. Tiny changes (brightness of plus or minus 0.1) barely matter.

**Before and after.** A dropdown picks one image from the sample and shows the original and the changed version side by side, each with the model's answer and confidence. The picture looks worse to a person, and the model often becomes more confident, not less. **Log the changed image** saves it to the failure log.

**What to try.** Move one slider at a time, from small to large, and write down the smallest setting that makes the table change noticeably. Find an image whose answer flips and log it.

**Common mistakes.**

- Moving three sliders at once and not knowing which one did the damage.
- Reading "predictions changed" as "accuracy dropped". A prediction can change from wrong to right. The first two table rows are the honest comparison.
- Logging an image where the model is still right. Ask whether it is actually a failure.
- Perturb always runs MobileNet live, whatever the precomputed switch says. With a graphics card it takes about a second. On a laptop without one, 100 images can take 10 to 15 seconds per change, so choose "50 images" and pause between moves.

#### Stamp (shortcut learning)

This is the shortcut lab. It has a "Why this matters" note and four numbered cards.

**1. Design the stamp.** Choose where the white "R" goes in the *training* images: **Pneumonia images** (default), **Normal images** or **Both classes**. "Both classes" puts the marker on everything, so it carries no information; that is the fix. You can also change Size (8 to 96 px, default 48), Corner (top-left default), Offset x and y, and Opacity. Two previews show "Without stamp" and "With stamp". **Use the default stamp** resets everything.

A grey line tells you which path you are on: "Default stamp + precomputed features: instant." or "Custom stamp: the images are processed live on this laptop (slower, with progress)." Any change from the default forces live feature extraction, which takes seconds on a laptop with a graphics card and minutes without one. A suggested challenge: find the smallest stamp that still fools the model (about 32 px, a seventh of the picture's width, in the default setup).

**2. Retrain on the stamped training set.** One button, **Stamp training set and retrain**, trains a new model with the same recipe as your normal model, except that augmentation is not used. Your normal model is kept as "Before". When it finishes: "Stamp-trained model ready: stamp on pneumonia images, 48 px, top-left, 20 epochs." **Discard the stamp-trained model** starts over.

**3. Flip rate.** Choose the "Test images" (Kermany test or Adult test) and "Stamp the TEST images of" (normal by default). The **flip rate** is the share of images whose answer changed once the stamp was added. Values on Kermany test with the stamp on the normal test images:

| Model | Said "pneumonia" without stamp | With stamp | Flip rate |
| --- | --- | --- | --- |
| Before: normal model | 46.2% | 63.7% | 17.5% (41 of 234) |
| After: stamp-trained model | 1.3% | 94.9% | 93.6% (219 of 234) |

The stamp-trained model turns about 94% of healthy children into pneumonia cases because a letter appeared in a corner. The normal model reacts a little too (about 18%); the comparison matters, not the zero.

A second table shows accuracy on the clean, unstamped images, and this is where shortcuts hide:

| Predictor | Accuracy | Sensitivity | Specificity | Balanced |
| --- | --- | --- | --- | --- |
| Before: normal model | 82.4% | 99.5% | 53.8% | 76.7% |
| After: stamp-trained model | 73.4% | 58.2% | 98.7% | 78.5% |

On clean data the shortcut model has a higher balanced accuracy than the honest one. The tell is its sensitivity: it misses about 42% of unstamped pneumonia, because it learned "marker present means pneumonia" instead of anything about lungs. Without the stamp test you would not notice.

**4. Try it on single images.** A grid of test thumbnails (12 per page, 24 per class). Click one to put the stamp on it, click again to remove it. A panel shows the picture with two answers, "Before (normal model)" and "After (stamp-trained model)", and a heatmap you can open for both. Stamp a normal X-ray and watch the "After" answer flip. **Log this** saves the case at once, tagged "shortcut" when the stamp is on and "other" when it is off, and an editor appears so you can change the tag.

**Common mistakes.**

- Changing the stamp size and then thinking the app has frozen. Use "Use the default stamp" to get back to the instant path.
- Stamping the *normal* training images and then testing with the stamp on normal test images. The model learned the opposite rule, so the flip test measures the wrong thing. Keep "Stamp the TEST images of" on the class that was *not* stamped in training.
- Retraining your normal model afterwards and reading the old flip table. It will say "The models changed since this was measured. Press 'Compute flip rate' again."
- Looking for the stamp-trained model on the Results tab. It is not there. The results card always says "Trained on stamped images: no".
- A stamp log entry shows the *normal* model's answer. The stamp-trained model's probability is stored only in the exported file, so put what it said in the hypothesis text.

#### Your images & webcam (closed world)

**Your images.** A red notice says not to upload your own or family members' medical images. A drop zone with **Choose images** takes JPEG, PNG or WebP files up to 10 MB each, at most 50 images. Each is cropped to a centered square and resized to 224 x 224. Every image gets a thumbnail, a label dropdown (**unlabeled**, **pneumonia**, **normal**) and a **Remove** button; **Clear all** removes everything.

"Predictions for your images" shows, for each image, your label, the model's answer with its confidence, and Log this and Heatmap buttons. If you labelled some images, a score table appears with "Scores use only the N images you labeled." Leave pictures that are not X-rays unlabeled, because a cup has no true label.

The point is stated on screen: "The model only knows two answers. Whatever you show it, it must pick pneumonia or normal (a closed world): it has no 'none of the above'." Good test pictures are a cup, a face, a landscape, a drawing. Obvious objects make the point better than X-rays of other body parts. In testing, a handful of non-X-ray pictures were all called pneumonia, most with 90% confidence or more.

**Live webcam.** **Start camera**, **Stop camera** and **Freeze & log**. The live feed shows an overlay such as "pneumonia, 95% sure", updated about four times a second. Point it at a cup or your hand: the model still picks a class. Move the refer slider to 90% while it runs and some frames switch to orange "Refer to a doctor". **Freeze & log** saves the frame to the failure log at once with the tag "other" and opens an editor; change the tag to closed-world and write a hypothesis.

The camera only works on `https://` or `localhost`, and only after you have trained a model. If permission is denied, click the camera or lock icon in the address bar, set Camera to Allow, reload and press Start camera again. "The camera is busy" means another app or tab (a video call, for example) is using it. The webcam is optional; uploads show the same thing.

### 2.7 Failure log

**What you see.** A "Counts by tag" card with a chip for each of the seven tags plus "all", a "What do the tags mean?" fold, an **Export JSON** button and a list of entries, newest first. Clicking a chip filters the list; clicking again clears the filter.

Each entry shows a thumbnail of exactly what the model saw (including a stamp or a slider change), a coloured tag, the source ("stamp", "uploads", "perturb (kermany_test)" and so on), the time, a line such as "Truth: normal. Model said: pneumonia, 95% sure (P(pneumonia) = 95.0%). Wrong.", your hypothesis, and the model settings in force. Each has **Edit tag and hypothesis** and **Delete**.

**How entries get in.**

| Where | Button | What happens |
| --- | --- | --- |
| Evaluate prediction list, uploads list | Log this | Opens a small form. Pick a tag, optionally type a hypothesis, then press "Save to failure log". |
| Break it, Perturb | Log the changed image | Same form. Saves the altered image. |
| Break it, Stamp, single images | Log this | Saves at once. Tag is "shortcut" with the stamp on, "other" without. An editor appears. |
| Break it, webcam | Freeze & log | Saves at once with the tag "other". An editor appears. |

The one-click routes fill the log with "other" if you never open the editor. The tag says what happened; the hypothesis says why. Both matter.

**Limits.** The log holds at most 100 entries and lives in the browser. If the browser refuses to store it (private mode, full storage), a notice warns that it only lives in this tab. **Export JSON** saves a file like `vision-failures-20260101-1015.json` with every entry, thumbnail and setting. Log this also works on correct predictions, which is fine if you mean it as a control.

Section 5 explains the seven tags and how to write a hypothesis.

### 2.8 Results

**What you see.** A control card, then the results card. Print prints only the card, which fits on one page.

**Control card.**

- **Evaluate all three sets now** scores any of Kermany test, Adult test and Kermany stress that have not been scored yet. It is greyed out when all three are done.
- **Re-run all three with the current settings** redoes all three, for example after flipping the precomputed switch.
- **Download results JSON** and **Print**.
- A status line such as "3 of 3 sets evaluated. Features: precomputed."

**Leaderboard value (worst-case score).** A large number with the set that set the minimum, the refer threshold (tau) and the coverage on that set. For the default model: **51.5%, Adult test, tau 50%, coverage 100%**. Until all three sets are scored it says "Not complete yet."

The worst-case score is the lowest of the three sets' half-credit scores. At tau 50% nothing is referred, so half-credit equals balanced accuracy, the average of sensitivity and specificity. That is why the stress set contributes 60.7% even though its accuracy is 70.4%, and why this number is not the accuracy you saw on Evaluate or Train. The idea is that a deployed model does not choose which patients arrive, so the score asks how bad things can get, not how good they can look.

**Scores per test set.** Two rows per set (your model and the baseline) with Accuracy, Sensitivity, Specificity, Balanced accuracy, Referred n (%), Coverage, Half-credit score and Features. The weakest set is tagged "(worst case)". When the slider is above 50% the first three columns say "(answered cases)".

**The model behind these numbers.** Balance mode, adult mix, augmentation, epochs, learning rate, seed, refer threshold and "Trained on stamped images: no". Check this block first when a number looks odd.

**Failures you logged.** The seven tag chips with counts.

**Snapshots.** A text box (suggested names "Before fix", "After fix 1", "After fix 2"), a **Save snapshot** button and a counter "N of 8 saved". Saving is disabled until all three sets are scored. Each snapshot becomes a column. Every column after the first shows the change in percentage points from the one before, in green or red. Each snapshot keeps the tau and settings it was saved with, so snapshots taken at different thresholds mix two effects.

Example, the default model against "balanced plus adult mix":

| | Before fix | After fix 1 |
| --- | --- | --- |
| Model | realistic, adult mix off | balanced, adult mix on |
| Worst-case score | 51.5% (Adult test) | 59.5% (+8.0 points, still Adult test) |
| Kermany test: accuracy / sensitivity / specificity / half-credit | 82.4 / 99.5 / 53.8 / 76.7 | 83.8 / 97.7 / 60.7 / 79.2 |
| Adult test: accuracy / sensitivity / specificity / half-credit | 51.5 / 90.0 / 13.0 / 51.5 | 59.5 / 41.0 / 78.0 / 59.5 |
| Kermany stress: accuracy / sensitivity / specificity / half-credit | 70.4 / 99.2 / 22.2 / 60.7 | 76.8 / 96.7 / 43.6 / 70.1 |

Look at what the fix traded away. Sensitivity on the adult set fell from 90% to 41%, while specificity rose from 13% to 78%. The score went up, but a sick adult is now missed more often than caught. A better score is not automatically a safer model.

**Common mistakes.**

- **Retraining before saving "Before fix".** Training again throws away the old model and its evaluations, so there is nothing left to snapshot. The order is: evaluate all three, save "Before fix", change the recipe, press Train again, evaluate all three, save "After fix 1".
- Changing the recipe on Data and expecting the card to change. Press Train again.
- Raising the refer threshold to climb the score. It does the opposite.
- Pressing Reset session to fix a small problem. It deletes the log and the snapshots.
- Comparing a precomputed run with a live-extraction run. They match to within a small amount, but the Features column says which one produced each number. If the switch was flipped after some sets were scored, a yellow note names them.

Changing the slider recomputes the card from stored predictions, with a note "Recomputed for the current threshold (tau = 90%)".

### 2.9 Workshop steps

A collapsed bar under the tab bar reads "Workshop steps (0 of 11 done)". Open it for two lists of checkboxes, headed Day 1 and Day 2:

- Day 1: load data and read the class counts; train your first model; compare with "always say pneumonia"; evaluate on the adult X-rays; break it three ways (sliders, upload, webcam on a cup); log each failure with a tag.
- Day 2: run the stamp experiment and record the flip rate; test on the adult set and note the drop; apply two fixes (balance, adult mix, augmentation, refer threshold); re-measure all three sets and save a snapshot; post your worst-case score.

Ticking is manual and nothing ticks itself. Ticks survive a reload. The list is a "where am I" card for the two-session path in section 4; ignore it if you work in your own order. The adult set appears in both lists on purpose: a quick look in the first session, the full shift lab in the second.

### 2.10 Settings drawer

Opens from the Settings button. Escape or Close leaves it.

- **Use precomputed features** (on by default). On, the app reads MobileNet's 1,280 numbers for each image from files shipped with the page, so training and evaluation take seconds. Off, MobileNet runs on every image on your computer, which gives the same numbers but takes minutes on a weak machine. You can flip it at any time; the next training or evaluation uses the new mode, and each result says which mode made it.
- **Backend** and **Model SHA-256** are information only. The hash lets two people confirm they have the same MobileNet file.
- **Reset session** (red, asks for confirmation). It clears the trained models, uploads, camera, failure log, snapshots, checklist ticks and saved settings, and puts the precomputed switch, refer threshold and training recipe back to their defaults. MobileNet stays loaded. Export the failure log first if you want to keep it.

---

## 3. Glossary

Numbers marked "about" are from the default run and move a point or two with the seed and backend.

| Term | Plain meaning |
| --- | --- |
| **Backend** | The part of the browser that does the number crunching: WebGL (uses the graphics card, fastest), WASM (a fast CPU path), or plain CPU (slowest). |
| **P(pneumonia)** | The model's output, a number from 0 to 1. At 0.5 or above it says pneumonia. |
| **Confidence** | How sure the model is of whatever it said: the larger of P and 1 minus P, so always between 50% and 100%. A confident wrong answer is common. |
| **Sensitivity** | Of the children who really have pneumonia, the share the model caught. About 99.5% on the Kermany test (388 of 390). |
| **Specificity** | Of the healthy children, the share the model correctly cleared. About 54% on the Kermany test; about 13% on the adult test. |
| **Accuracy** | The share of all cases called correctly. Misleading when classes are unequal: always answering "pneumonia" gets 62.5% on the Kermany test. |
| **Balanced accuracy** | The average of sensitivity and specificity. A model that ignores the picture scores 50%. About 77% on the Kermany test. |
| **Confusion matrix** | A 2 by 2 table counting right and wrong answers for each true class: correctly cleared, false alarm, missed, caught. |
| **Baseline** | A pretend model that never looks at the picture and always says "pneumonia". It scores about 62.5% on the Kermany test and 50% on the adult test. Your real model must beat it. |
| **Class weights** | How much each class counts when the training error is added up. Realistic: 1.00 and 1.00. Balanced: normal 2.00, pneumonia 0.67. |
| **Augmentation** | Adding randomly brightened, darkened, rotated or cropped copies of each training image (two extra views per image) so the model learns those changes are not the disease. |
| **Training set** | The 1,020 images the model learns from. |
| **Validation set** | Images used while training to watch progress. Ours has 180 images from the same pool as the training set, so it is optimistic: about 99%. |
| **Test set** | Images the model never saw while training. The Kermany test has 624 and gives the honest score for the same kind of patient: about 82%. |
| **Adult test** | 200 adult X-rays from another hospital, 100 pneumonia and 100 no finding. About 52% accuracy for the default model, barely above the 50% baseline. |
| **Stress set** | The same 624 Kermany test X-rays, made darker, lower in contrast, rotated 5 degrees and cropped to 95%. About 70% accuracy, about 61% balanced accuracy. |
| **Distribution shift** | The new data looks different from the training data (another age group, hospital or machine) and the score drops. Here from about 82% to about 52%. |
| **Perturbation** | A deliberate change to a test picture: brightness, contrast, rotation or crop. |
| **Shortcut learning** | The model uses a cue that predicts the label in the training data but is not the disease, such as a stamped "R". |
| **Stamp** | The white "R" the app draws in a corner of selected images. |
| **Flip rate** | Of the images that received the stamp, the share whose answer changed. About 94% for the stamp-trained model on normal test X-rays, about 18% for the normal model. |
| **Closed world** | The model only knows two answers, so it must pick one for anything, including a cup. |
| **Abstention / referral** | The model declines to answer when its confidence is below the threshold on the slider. The app calls this "Refer to a doctor". |
| **Tau** | The name of that threshold. 50% means never refer. |
| **Coverage** | The share of cases the model answered. About 75% at tau 90% on the Kermany test. |
| **Half credit** | The scoring rule for the leaderboard: a referred case counts as half right, so referring everything scores exactly 50. |
| **Worst-case score** | The lowest half-credit score of the three test sets (Kermany test, Adult test, Kermany stress). About 52 for the default model, set by the adult test; about 59 with balanced classes plus the adult mix. |
| **Snapshot** | A saved copy of all three sets' scores and the model settings, so before and after sit side by side. Up to 8. |
| **Seed** | The starting point for the random numbers. The same seed and settings give the same model. Default 42. |
| **Epoch** | One pass over all the training images. Default 20. |
| **Learning rate** | The size of each correction step during training. Default 0.001. |
| **Loss** | The error number that training shrinks. Lower is better. |
| **Precomputed features** | MobileNet's 1,280 numbers per image, calculated once beforehand and shipped as files so you skip the slow step. Same numbers as live extraction. |
| **Live extraction** | Running MobileNet on every image on your own computer. Slow but gives the same result. |
| **MobileNet v2** | The frozen network that turns an image into 1,280 numbers. |
| **Head** | The small trainable layer on top of MobileNet. |
| **Grad-CAM** | A way to colour the parts of an image that pushed the model towards its answer. Here the map is 7 by 7 cells, so it is a rough hint. |
| **Failure log, tag, hypothesis** | The record of a failure, the category of why it failed, and your testable guess about why. |

---

## 4. Suggested exercises

The two sessions below work for a class or for one person on their own. Each is about one and a half to two hours, but they can be cut up or spread out. They are an outline of tasks and questions, not a script.

**About the numbers.** The figures here come from the default run: seed 42, 20 epochs, realistic training, precomputed features. They match from laptop to laptop for the same seed and backend. With a different seed or backend they move a point or two. Over 20 seeds the Kermany test accuracy ranged from about 79% to about 83%, so seed 42 sits at the top of that range. If your numbers are close, nothing is wrong.

Typical values to expect:

| What | About |
| --- | --- |
| Validation accuracy (180 images) | 99% |
| Kermany test accuracy | 82% (sensitivity 99.5%, specificity 54%) |
| Adult test accuracy | 52% |
| Kermany stress accuracy | 70% (balanced 61%) |
| Stamp flip rate, stamp-trained model | 94% |
| Worst-case score | 52, rising to 59 after balanced classes plus the adult mix |

### Session 1: Train it and doubt it

Goal: build a model, see a flattering score, then find out why not to trust it.

1. **Look at the data (Data).** Read the count table. Write down the ratio of pneumonia to normal in training. Browse the gallery and try to tell the classes apart. *Question: if the model always said "pneumonia", what accuracy would it get on the test set?* (390 of 624, or 62.5%.)
2. **Train the first model (Train).** Keep the defaults. Note the validation accuracy. *Question: is 99% good?* The right reply is "compared with what?"
3. **Take the score away (Evaluate).** Open Kermany test. Write down accuracy, sensitivity and specificity, and the baseline row. Find the 108 false alarms in the confusion matrix. *Questions: which mistake is worse here, a miss or a false alarm? Why does sensitivity look great while specificity does not?*
4. **A first look at other sets (Evaluate).** Switch to Adult test and Kermany stress. Record the accuracy for each. *Question: which number would you quote to a hospital, 99, 82 or 52?*
5. **Predict, then break (Break it).** Before opening the tab, write one sentence: "It will fail when ______ because ______." Then try to prove yourself wrong with each of these, one at a time:
   - Perturb: move Brightness to -0.40, then try Rotation 20 degrees and Crop 70%. Note the smallest change that makes the table move.
   - Your images: upload a cup, a landscape and a page of text, or point the webcam at a cup. *Question: did the model ever say "I don't know"? Why can't it?*
   - Optional for fast groups: Train tab, Repeat x5. Copy the three spread numbers.
6. **Log what you found (Failure log).** Log at least one failure per experiment, each with a tag and a hypothesis. Read the tag counts. *Question: which tags are still empty, and what would you have to do to fill them?*

### Session 2: Shortcuts, shift and fixes

Goal: find the two failures that cannot be seen on a normal test set, then try to repair the model and measure what the repairs cost.

1. **Recap (Failure log).** Look at the tag counts from the first session. Retrain with the defaults if your model was lost; the same seed reproduces the same numbers.
2. **The shortcut lab (Break it, Stamp).** Leave the stamp at its defaults. Press "Stamp training set and retrain". Record the flip rate for both models. Read the clean-accuracy table and compare balanced accuracy with sensitivity. Click a normal X-ray in step 4 and watch the "After" answer flip. Open the heatmap. *Questions: why did a model this wrong score so well on the exam? Where could a real marker come from? How would you fix it?* Then try "Both classes", retrain and measure the flip rate again. Challenge: find the smallest stamp that still fools the model.
3. **A new population (Evaluate, Results).** Open Adult test and note the drop from the Kermany test. List two visible differences between the datasets. Then go to Results, press "Evaluate all three sets now", and **save a snapshot called "Before fix"**.
4. **Fix it (Data, Train, Results).** Tick at least two fixes: Balanced, "Mix in adult X-rays", Augmentation. Press Train again with the same seed. Evaluate all three sets and save "After fix 1". Repeat with a different combination. What you should see:

   | Step | Kermany test (half-credit) | Adult test | Kermany stress | Worst case |
   | --- | --- | --- | --- | --- |
   | Realistic, no fixes | 76.7% | 51.5% | 60.7% | 51.5% |
   | Balanced only | 79.9% | 51.5% | 64.9% | 51.5% |
   | Balanced plus adult mix | 79.2% | 59.5% | 70.1% | 59.5% |
   | Plus augmentation | 77.0% | 59.0% | 78.5% | 59.0% |

   Balanced classes alone leave the worst case at 51.5% because the adult set is the weakest one and class weights do not help it. Only the adult mix lifts the worst case. Augmentation repairs the stress set a lot but costs the other sets a little. A fix can help one set and hurt another.
5. **Try the refer slider.** Set it to 90% and watch Referred and Coverage appear. Check the worst-case score on Results. *Questions: is the model better, or just quieter? Why did accuracy go up while the score went down?*
6. **Check the cost.** Compare adult sensitivity before and after the adult mix (about 90% to about 41%). *Question: your score went up; did the model get safer?*
7. **Report the result.** Write down the worst-case score, the set that set it, the seed, the fixes used and the refer threshold. Compare with someone else. A difference of a point or two is a tie.

A closing note worth making: you picked fixes by watching the same test sets you report on. That is a small version of the dishonest-evaluation problem, and the failure log has a tag for it.

---

## 5. How to read a failure and write a hypothesis

Every logged failure gets one of seven tags. The tag follows the *cause*, not what the mistake looks like. A healthy child called "pneumonia" can be imbalance, quality or shift depending on what you did to get there:

| Where you saw it | What you see | Tag |
| --- | --- | --- |
| Kermany test, original picture | The model leans towards pneumonia | imbalance |
| Kermany stress, same picture, darker and tilted | It was right before, wrong now | quality |
| Adult test, a different person entirely | Wrong again, more often | shift |

The cause decides the fix: class weights, augmentation or adult X-rays.

| Tag | Meaning | One plain example |
| --- | --- | --- |
| imbalance | One class is much more common in the training data, so the model leans towards it. | On the Kermany test, a healthy child is called pneumonia with 96% confidence, and the training set had three times as many pneumonia images. |
| shortcut | The model reacts to a marker or artefact (a stamp, a label, a corner) instead of the lungs. | A normal X-ray with the stamp on is called pneumonia by the stamp-trained model. |
| shift | The image comes from a different population or machine than the training images. | An adult with no finding is called pneumonia on the adult test. |
| quality | The picture is darker, lower in contrast, tilted or badly framed compared with the clean training images. | At brightness -0.40, a child's X-ray that was answered correctly is now called pneumonia with 95% confidence. |
| closed-world | The model must answer pneumonia or normal even when the picture shows something else. | A drawing of a cup is called pneumonia, 83% sure. |
| evaluation | The score itself misleads: too few test images, the wrong metric, or test data that leaked into training. | The 16-image validation score jumps between 81% and 87% across seeds; or you read "82% accuracy" while specificity is 54%. |
| other | None of the above, or not sure yet. Say what you are unsure about in the hypothesis. | A mistake nobody can explain yet. |

### Telling similar tags apart

- **Shift or quality?** Imagine the same patient scanned again on the training machine. If the model would be right, it was quality. If it would still be wrong because adults, hospitals and machines simply look different, it was shift.
- **Quality or shortcut?** Quality is a general worsening of the picture. A shortcut is a specific cue that goes with the label in the training data. A stamped image is not a worse picture; it is a picture with a cue.
- **Shift or closed-world?** A shift picture is still a chest X-ray. A closed-world picture is not one.
- **Imbalance or shift?** Imbalance happens on the same kind of patient as the training set. Shift needs a different population.
- **Evaluation** is about the number, not the picture. Every other tag is about an image. The Repeat x5 card has no Log this button, so log any Evaluate row and change its tag to "evaluation" with a hypothesis about the metric.

### Writing a hypothesis

The tag says what happened. The hypothesis says why you think it happened, in a way you could test. A useful template:

> I think it failed because ______. Evidence: ______. Test: ______.

| Quality | Example |
| --- | --- |
| Good | "I think it failed because the picture is darker than the training images. Evidence: the original is answered correctly and the darkened copy is called pneumonia with 95% confidence. Test: retrain with augmentation on and see whether the number of changed answers falls." |
| Good | "I think it flagged this healthy child because the model leans towards pneumonia. Evidence: 108 of 234 healthy children were flagged and the training set is 1 to 3. Test: switch to Balanced and see whether the false alarms drop below 108." |
| Weak | "The AI is bad at dark pictures." (No evidence, no cause, no test.) |
| Weak | "It is not trained enough." (Hard to test, and epochs change very little here.) |

A good question to ask yourself: what change could I make that would prove me wrong?

One image is an anecdote. If the mistakes "look dark", count how many of the first 20 mistakes look dark and how many of the first 20 correct answers do. 12 against 3 means something; 8 against 7 does not. Then confirm with a controlled change such as the brightness slider. Always look at the correct ones too, and treat a heatmap as a hint, not an explanation.

---

## 6. Troubleshooting, URL parameters and data sources

### Troubleshooting

| Symptom | Likely cause | What to do |
| --- | --- | --- |
| Blank page, or stuck on "Loading" | Old cache, blocked files or a very old browser | Hard reload (Ctrl+Shift+R). Try a current Chrome, Edge or Firefox. If you run it from a local copy, serve it through `localhost` rather than opening `index.html` directly. |
| Header says `No backend`, or a red "No compute backend available" banner | The browser could not start any compute backend | Try another browser, or add `?backend=wasm` to the address. |
| Backend badge is yellowish and says WASM or CPU | No usable WebGL | Fine. Training is still seconds with precomputed features. Perturb and live extraction are slower; use the 50-image sample. |
| "WebGL context lost" banner | The graphics driver reset, often with many tabs open | Reload, or open with `?backend=wasm`. |
| Camera does not open | The page is not `https://` or `localhost`, permission was denied, or another app has the camera | Use the hosted link or `localhost`. Click the lock icon in the address bar, set Camera to Allow, reload. Close video calls. Uploads show the same thing. |
| Training takes minutes ("Running MobileNet on image 200 of 1200") | The precomputed switch is off, or the feature files failed to load | Turn "Use precomputed features" on in Settings. On the Data tab the dot should say "Using precomputed features". If you see "Precomputed features could not be loaded", press Try again. |
| Training stretches out for no reason | You are on another browser tab, which slows the page | Stay on the app tab until it finishes. |
| Stamp lab is slow | A custom stamp forces live extraction | Press "Use the default stamp" and keep the precomputed switch on. |
| A Perturb slider takes 10 to 15 seconds | Perturb always runs MobileNet live | Choose "50 images". Pause between slider moves. |
| Tab is sluggish after a long session | Accumulated state | Export the failure log, then use Reset session. |
| Stuck mid-run | A cancelled or failed job left the page odd | Reload. The trained model and the log come back. If it is still odd, use Reset session. |
| Your numbers differ from someone else's | Different seed, recipe, precomputed switch, refer threshold, or a model restored from an earlier visit | Compare the "Model in use" strip and "The model behind these numbers" on Results. Set the seed to 42, the threshold to 50%, the switch on, and press Train again. Live extraction can differ from precomputed by a small amount. |
| Results says "Not complete yet" | One or more of the three sets has not been scored | Press "Evaluate all three sets now". |
| Results numbers vanished after retraining | Training again discards stored evaluations | Evaluate all three sets again. |
| The flip table says "The models changed since this was measured" | You retrained the normal model after the stamp model | Press "Compute flip rate" again, or retrain the stamp model. |
| Failure log is full or shows a storage warning | The log holds 100 entries, or the browser blocks storage | Export JSON, then delete entries. Export before closing the tab if storage is blocked. |
| Reloading offline fails | The page needs to be loaded once | Once loaded, the app needs no network. A reload without any connection needs a locally served copy. |

### URL parameters

Add these to the end of the web address. Combine them with `&`, for example `?backend=wasm&debug=1`.

| Parameter | Effect |
| --- | --- |
| `?backend=wasm` | Use WebAssembly instead of WebGL. The backend badge tooltip says it was forced. |
| `?backend=cpu` | Use the plain CPU. Slowest, a last resort. |
| `?backend=webgl` | Prefer WebGL. The app still falls back by itself if WebGL cannot start. |
| `?debug=1` | Shows a small box in the bottom right with a memory count (tensors and bytes). Useful for developers. |

### Where the data comes from

- **Children's chest X-rays:** Kermany et al. 2018, Guangzhou Women and Children's Medical Center, ages 1 to 5, licensed CC BY 4.0. The official test set is used unchanged (234 normal, 390 pneumonia), plus a seeded sample of 1,200 images from the official training split.
- **Adult chest X-rays:** NIH ChestX-ray14, taken through MedMNIST v2 (CC0). The test and optional training sets are hand-balanced 50/50 because pneumonia is rare in the original data.
- **Feature extractor:** MobileNet v2, trained on ImageNet (Apache 2.0).

Both X-ray collections are published, anonymised research datasets. The adult labels were extracted from reports by software, so some are wrong. The children's data has no patient ids, so a child may appear more than once across splits, which makes the validation score slightly optimistic. The official test set contains 6 pixel-identical duplicates. Full citations, licences and limitations are in [DATA_LICENSES.md](../DATA_LICENSES.md).

---

## 7. For instructors

You can reuse the app in your own course. A few practical points.

**It is a static site.** The built folder (`app/dist`) contains the page, the model, the images and the WebAssembly files, and makes no external requests. Serve it from any web server, from a school intranet, or from a USB stick with the bundled helper:

```bash
cd app
npm ci
npm run build
node dist/serve-offline.mjs      # http://127.0.0.1:8080
```

`npx serve dist` and `python -m http.server` in the `dist` folder also work. Serve through `localhost` or `https://`, not by double-clicking `index.html`, because browsers block model loading from `file://` and only allow the camera on secure pages. The README covers the full build and deployment steps.

**Results are repeatable.** The default seed is 42, the data splits are fixed by a seeded script, and the precomputed features are shipped as files. Everyone using the same settings and the same backend gets the same numbers, give or take a point on a different backend. That lets a whole room compare results, or lets you check your own against the figures in this guide. Change the seed to show that numbers move.

**Before a session, check:**

- The header shows `Ready` and the backend you expect. Press Train and open Evaluate on all three sets. You should see about 82%, 52% and 70%. Then use Reset session so you start clean.
- Allow the camera once so the permission prompt is not a surprise, and have `?backend=wasm` ready for laptops with poor graphics support.
- A laptop that was used before opens with a restored model. That is convenient but can hide the training step, so reset when you want a fresh start.
- Do not reset between sessions if you want to use the failure log counts for a recap.

**Pacing tips.** The Perturb tab is the slowest part on laptops without a graphics card; the 50-image sample helps. Tell students to stay on the app tab while it trains. Remind them to save "Before fix" before retraining. If time is short, cut the Repeat x5 demo and keep the baseline comparison, or cut the refer slider and keep the stamp lab.

**Pairing works well.** Two people per laptop, one driving and one writing the hypothesis, swap halfway. Give pairs different starting tabs so the failure log is varied.

**Adapting the app.**

- *Swap the stamp.* The marker is drawn by shared image code used both by the app and by the script that precomputes features. Change it in `app/src/shared/pixelOps` and regenerate the features with `scripts/precompute_embeddings.mjs`, otherwise precomputed and live results will disagree.
- *Change the datasets.* The preparation scripts (`scripts/prepare_kermany.py`, `scripts/prepare_adult.py`) build the sample sizes, the splits and the 50/50 adult balance from seeded sampling. Edit the counts or the sources, rerun them, and rebuild the features. `scripts/tests/verify_data.py` and `scripts/tests/verify_model.mjs` check the output.
- *Change the stress preset or the defaults.* The stress degradation and the default training settings live in the app source; the numbers in this guide will shift accordingly.
- *Use the code track.* `notebook/code_track.ipynb` repeats the experiments in Python for students who want to see the code behind the buttons.
- *Slides.* The `Slides/` folder has a deck that covers the background ideas, and `Screenshots/` has images from the app.

**A caution for any audience.** The app deliberately shows a model that looks good and is not. Say clearly that this is a teaching exercise: the model is trained on a small sample from one hospital, and nothing here is evidence about real clinical performance.
