# Teaching guide: Break it, the Failure log and Results

This guide is for you, standing alone in front of a room of mostly beginners. You told me these three tabs are the core thing to explain, so it goes deeper on them than the UI guide does. Read `docs/ui-guide.md` for the tab-by-tab reference and `docs/dry-run-results.md` for the raw measurements. This one is about how to teach.

Educational only. Not for clinical use. Say it at the start of each day.

A note on numbers. Everything below is from the default run: seed 42, 20 epochs, realistic 1:3 training, precomputed features, refer threshold at 50%. Your laptop should match to about a point. A student who changes the seed will see different numbers. The Kermany test accuracy ranged from 79.2% to 82.7% over 20 seeds (mean 80.9%), and seed 42 sits at the top of that range. So on stage say "about 81". Where I write "about", the number moves with the seed or the backend. Where I write a number with a decimal, it is what the app shows for seed 42.

The four numbers to keep in your head, because everything else hangs on them:

| Set | Accuracy | Sensitivity | Specificity | Balanced accuracy |
| --- | --- | --- | --- | --- |
| Kermany test (624 children) | 82.4% | 99.5% | 53.8% | 76.7% |
| Adult test (200 adults) | 51.5% | 90.0% | 13.0% | 51.5% |
| Kermany stress (same 624, degraded) | 70.4% | 99.2% | 22.2% | 60.7% |
| Validation on our 180 images (Train tab) | 98.9% | | | |

The model is good at saying "pneumonia" and poor at saying "normal". Every break below is a different way of seeing that same weakness, or of seeing a new one.

---

## 1. The idea behind "Break it"

### Why break a model we just trained

After the Train tab, every student holds a model with a validation accuracy near 99%. It feels finished. The point of Break it is to replace that one number with a list of reasons the model can be wrong, each of which has a name and a fix.

You are not teaching students that AI is bad. You are teaching them that a score is the answer to a question, and that a different question gets a different answer. "How does it do on children from the same hospital?" gets 82%. "How does it do on adults?" gets 51%. Neither is a lie. Both are needed before anyone trusts it.

There is a second reason. Beginners cannot judge a model by reading code or loss curves, but they can judge a picture. When a student sees a healthy child's X-ray called pneumonia with 96% confidence, they understand the problem faster than from any slide. Breaking it is how the abstract ideas (imbalance, shift, shortcut) become something they have seen with their own eyes.

### The six breaks and where they live

Only three of the six live under the tab called Break it. The others are on Evaluate and Train. Students will notice, so say it early: "Break it is a way of working, not just a tab."

| Break | Where in the app | Exact clicks | What students see (seed 42) | What it teaches | Failure-log tag | The fix |
| --- | --- | --- | --- | --- | --- | --- |
| Imbalance | Data, then Evaluate | Data: read 255 normal and 765 pneumonia. Train. Evaluate, Kermany test. Read the "Always say pneumonia" row, then the confusion matrix. | Baseline 62.5%. Model 82.4% accuracy, but specificity 53.8%: 108 of 234 healthy children flagged. The model says "pneumonia" for about 80% of all images. | A common class pulls the model towards it. Accuracy flatters it. | imbalance | Data tab, Class balance: Balanced. Specificity goes to 61.5% and false alarms drop to 38.5%. Worst-case score does not move (51.5%). |
| Shortcut | Break it, Stamp | Train first. Stamp tab, leave the defaults, click Stamp training set and retrain. Read the flip table, then the clean-accuracy table. Then step 4: click a normal X-ray to stamp it. | Stamp-trained model flips 93.6% (219 of 234) of normal test X-rays to pneumonia. The normal model flips 17.5% (41 of 234). On clean images the stamp model has sensitivity 58.2% and balanced accuracy 78.5%, higher than the honest 76.7%. | The model uses the easiest cue that predicts the label. A good test score can hide this. | shortcut | Put the stamp on both classes (Both classes) or remove the marker. Check the heatmap. |
| Shift | Evaluate | Evaluate, Test on: Adult test. | Accuracy 51.5% against a 50% baseline. Sensitivity 90%, specificity 13%: 87 of 100 no-finding adults flagged. A 30.9 point drop from the Kermany test. | The model only knows the kind of patient it trained on. | shift | Mix in adult X-rays (Data tab, Day-2 fixes). Worst-case score goes from 51.5% to 59.5%, but adult sensitivity falls from 90% to 41%. |
| Image quality | Break it, Perturb; also Evaluate, Kermany stress | Perturb: drag Brightness to -0.40, wait for the table, look at Before and after. Try Rotation 20 and Crop 70%. Then Evaluate, Kermany stress. | Brightness -0.40 on the 100-image sample: balanced accuracy 71.6% to 61.4%, 8 of 100 answers change. Rotation 20 degrees: 52.7%. Crop 70%: 50.0% (a coin flip). Stress set: 70.4% accuracy, 60.7% balanced. | Tiny changes do little, big changes do real damage, and the model does not get less sure when the picture gets worse. | quality | Augmentation (Data tab). Stress score rises from 70.1% to 78.5% once it is added on top of the adult mix. |
| Closed world | Break it, Your images & webcam | Choose images (a cup, a landscape, a text page) or Start camera, point at a cup. Optionally move the Refer to a doctor slider to 90%. | Six non-X-ray pictures: all six called pneumonia, four at 90% or more. A 90% threshold refers only two of the six (the noise and the cup). | The model has two answers and must give one. It cannot say "this is not an X-ray". | closed-world | Only a partial one in the app: the Refer to a doctor slider. The real fix is a human, or a separate check that the picture is a chest X-ray. |
| Dishonest evaluation | Train, Repeat x5 with new seeds | Train tab, scroll to the Repeat x5 card, click Repeat x5 with new seeds. About 6 seconds on a GPU. Read the Spread row. | Over seeds 1 to 5: the 16-image validation score spans about 6 points (81.3% to 87.5%), the 180-image score about 0.6 points, the Kermany test about 2 points. | A score from 16 images is noise. One image is worth 6.25 points. | evaluation | A bigger validation set (our 180 images) and a test set the model never touched. |

Two things to say about this table:

- The tag follows the cause, not the look of the mistake. "A healthy child called pneumonia" can be imbalance (Kermany test), shift (adult test) or quality (stress set). You find out which by running the experiment. That is the lesson of the whole Failure log, and section 4 returns to it.
- Refer to a doctor is a safeguard, not a sixth fix for the score. In the dry run it lowered the worst-case score (59.0% to 56.0% at 80%). Tell students that abstaining buys safety, not leaderboard points.

---

## 2. Break it, one section per experiment

Every Break it experiment needs a trained model. Before you start, check that the room has pressed Train and that the Evaluate tab shows 82.4% on the Kermany test (or about 81% for students who changed the seed).

The "Refer to a doctor when the model is less than X% sure" slider appears on both Evaluate and Break it, and it is one shared setting. Leave it at 50% for the first pass and tell students to leave it alone until you introduce it.

### 2.1 Perturb (image quality)

**Demo in two minutes.**

1. Break it, Perturb. Leave Test images on Kermany test and Sample on 100 images.
2. Drag Brightness to -0.40. A bar reads "Recomputing 100 images..." and then a table appears.
3. Read the three rows aloud: "Our model, with your changes", "Our model, original images", and "Always say pneumonia". The original row is the control. Balanced accuracy goes from about 71.6% to about 61.4%.
4. Read the line under the table: "8 of 100 predictions changed". Then scroll to Before and after and pick an image where the answer flipped. Show the two pictures side by side with their confidence.
5. Reset all. Drag Rotation to 20 and watch balanced accuracy fall to about 52.7%.

The thing to land: the picture looks worse to a person, and the model answers anyway, often with more confidence.

**What students do.** Move one slider at a time, from small to large. Write down the smallest setting that makes the table change noticeably. Find one image in Before and after whose answer changes, press Log the changed image, tag it, and write a hypothesis.

**What the numbers say.** Small changes barely matter. Brightness of plus or minus 0.1 changes 0 or 1 answers out of 100. Rotation of 3 degrees changes 5. Big changes hurt: brightness +0.4 takes balanced accuracy to 54.1%, and a 70% crop to 50%. Note that brighter is worse than darker (54.1% against 61.4% at 0.4), which surprises people.

**Questions for the room.**

- "What would a radiologist do with a very dark scan?" (Ask for a retake.)
- "What did our model do?" (It answered, and sometimes more confidently.)
- "Which slider would you worry about in a real clinic, and why?" (Rotation and crop are about positioning; brightness is about the machine.)

**Typical wrong turns.**

- Moving three sliders at once, then not knowing which one did the damage. Insist on one at a time.
- Moving a slider while a run is going. This is fine (the old run is cancelled), but on a laptop without a GPU each update takes 10 to 15 seconds and students think it has frozen. Choose "50 images" for those machines.
- Logging a changed image where the model is still right. The Log button saves whatever you are looking at, right or wrong. Ask: "Is this actually a failure? Did the answer change, and did it become wrong?"
- Reading "predictions changed" as "accuracy dropped". A prediction can change from wrong to right. The two rows above the sentence are the honest comparison.
- Expecting the first slider move to be fast. The first run loads the sample and is slower.
- Choosing the Adult test or Your uploaded images in the Test images dropdown and then wondering about the sample. With uploads the Sample menu is disabled and only labelled images get scores.

### 2.2 Stamp (shortcut)

This is the strongest demo in the workshop, and it belongs to Day 2. Keep the stamp at the default (48 px, top-left, on pneumonia training images) for the main run. Any change to size, corner, offsets or opacity switches the app to live feature extraction, which is fast on a GPU laptop (about 10 seconds per size) and slow without one (minutes for six sizes).

**Demo in two minutes.**

1. Break it, Stamp. Read the blue note "Why this matters". Point at the two previews, Without stamp and With stamp. The "R" is small. Say that.
2. Click Stamp training set and retrain. It takes about 2 to 3 seconds. The status reads "Stamp-trained model ready: stamp on pneumonia images, 48 px, top-left, 20 epochs".
3. Step 3, the flip table, with Test images on Kermany test and "Stamp the TEST images of" on normal. Read the Normal row of each model.

| Model | Said "pneumonia" without stamp | with stamp | Flip rate |
| --- | --- | --- | --- |
| Before: normal model | 46.2% | 63.7% | 17.5% (41 of 234) |
| After: stamp-trained model | 1.3% | 94.9% | 93.6% (219 of 234) |

4. Read the next table, "Accuracy on the clean (unstamped) images". The stamp model has accuracy 73.4%, sensitivity 58.2%, specificity 98.7%, balanced accuracy 78.5%. Pause on the balanced figure, which is higher than the honest model's 76.7%. Ask: "If you only saw this table, which model would you ship?" Then point at sensitivity: it misses about 42% of unstamped pneumonia.
5. Step 4, Try it on single images. Click a normal X-ray to put the stamp on it. Watch "After (stamp-trained model)" flip to pneumonia. Click again to remove the stamp and watch it flip back. Press Show heatmap and compare the Before and After maps.

**What students do.** Run steps 2 and 3 and write down the flip rate for both models. Do the single-image toggle on three different X-rays. Log one stamped image. Fast finishers: the smallest-stamp challenge, then Both classes.

**The smallest-stamp challenge.** Measured flip rate of the stamp-trained model on the 234 normal test images:

| Stamp size | 8 px | 12 px | 16 px | 24 px | 32 px | 48 px |
| --- | --- | --- | --- | --- | --- | --- |
| Flip rate | 6.4% | 10.3% | 12.8% | 27.8% | 53.4% | 93.6% |
| Normal model, same stamp | 1.3% | 2.6% | 3.4% | 5.6% | 13.7% | 17.5% |

So the smallest stamp that gets past 50% is about 32 px, roughly one seventh of the picture's width. Below about 16 px the effect is small. Tell fast finishers to compare the stamp-trained flip rate with the normal model's, not with zero.

**Both classes.** Choosing Both classes puts the marker on every training image, so it carries no information. I did not measure the flip rate after retraining this way, so try it once before the session. The expectation is that the flip rate falls a long way, but do not quote a number.

**Questions for the room.**

- "The stamp model scored a higher balanced accuracy than the honest one. Why did nobody notice?" (The exam images had no stamps.)
- "Where could a real marker come from?" (A scanner label, a hospital tag, a portable machine's pattern. Many of these X-rays already carry small R or L letters.)
- "How would you check a model for this without knowing the marker?" (The heatmap, and the flip test with a marker you invent.)

**Typical wrong turns.**

- Changing the stamp size, seeing a long progress bar, and thinking the app has hung. Use Use the default stamp to return to the instant path.
- Stamping the training images of "Normal images" and then testing with the stamp on normal test images. The model learned the opposite rule, so the flip test measures the wrong thing. Keep the "TEST images of" menu on the class that was not stamped in training.
- Expecting the stamp-trained model on the Results tab. It is not there. Results always reports the main model and says "Trained on stamped images: no".
- Retraining the normal model after the stamp lab, then reading the flip table. The table says "The models changed since this was measured. Press Compute flip rate again."
- The normal model also reacts to the stamp (17.5%). Students sometimes take that as a bug. It is the comparison that matters, 17.5% against 93.6%.
- Logging from step 4 without editing the tag. A stamped image is saved as "shortcut" automatically, an unstamped one as "other". Also see the note in section 4: the saved card shows the normal model's answer, not the stamp model's.
- Single-image view shows only the first 24 images of each class, 12 per page. That is enough for the demo.

### 2.3 Your images and webcam (closed world)

**Demo in two minutes.**

1. Break it, Your images & webcam. Read the red notice aloud: do not upload your own or family members' medical images.
2. Choose images and add a cup, a landscape and a page of text. Leave them unlabeled. Under "Predictions for your images", read the answers. In the dry run every non-X-ray picture was called pneumonia: a generated landscape 93%, a drawn cup 83%, a flat red square 100%, a page of text 100%.
3. Start camera. Hold up a cup. The badge says a class, such as "pneumonia" with a percentage, about four times a second. Press Freeze & log.
4. Move the Refer to a doctor slider to 90%. Some pictures turn orange and say "Refer to a doctor". In the dry run this caught the noise and the cup but let four of the six through confidently.

The sentence to say: "It had two answers and we gave it a third kind of thing. It did not say I don't know. It could not."

**What students do.** Upload three or four things that are not X-rays. Try the webcam on a cup, their hand and their face. Freeze & log a few frames, change the tag to closed-world and write a hypothesis.

**Questions for the room.**

- "Did the model ever say 'I don't know'? Why can't it?" (There is no third output.)
- "A confident answer to a cup. What does confidence actually measure?" (How far the number is from 50%. Not whether the picture is a sensible input.)
- "The threshold caught the cup but not the red square. Would you trust it?" (No. Some non-X-rays get 100%.)

**Typical wrong turns.**

- Labelling the cup "normal" or "pneumonia". The app then counts it in the scores table. A cup has no true label, so leave these unlabeled.
- Using real X-rays of other body parts. The point lands best with obvious objects.
- Freeze & log saves immediately with tag "other". Students forget to change it. Remind them. The log card will say "Truth: unknown", and it will not be marked "Wrong", because there is no right answer.
- The camera needs https or localhost. If it does not open, the uploads show the same thing.
- A student uploads a picture of themselves or their family. Say the rule on the slide, not only in the app.

### 2.4 The refer threshold

Introduce this on Day 2, after the stamp lab and the adult test. By then students have seen confident wrong answers, so a threshold feels like a natural response.

**Demo in two minutes.**

1. Evaluate, Kermany test, threshold at 50%. Note accuracy 82.4%.
2. Drag the slider to 90%. Two new columns appear, Referred and Coverage. Accuracy goes to 90.4%. The matrix caption reads "467 X-rays (157 referred, not counted)". Referred is 157 (25.2%), coverage 74.8%.
3. Say: "The model got better. Or did it get quieter?"
4. Results tab: the half-credit score on the Kermany test is 74.6%, lower than the 76.7% at 50%. Accuracy rose and the score fell.
5. Back to Evaluate, tau at 90%: click Only referred and look at the cases it handed back.

**The mechanism.** The model always has a lean (pneumonia if P is 50% or above). Its confidence is the larger of P and 1 minus P, so between 50% and 100%. The slider sets a bar. Below the bar the app does not answer. That is all. It does not make the model more accurate. It makes it say less, and the cases it keeps are the easier ones.

**Half credit.** A referred case counts as half right. Refer everything and you score exactly 50, the same as always saying pneumonia. That is deliberate: abstaining cannot game the leaderboard.

**Questions for the room.**

- "If the model refers a quarter of the cases, who looks at them, and are there enough doctors?" (This is a real cost.)
- "Which would you rather have, a model that is right 90% of the time on three quarters of the cases, or one that is right 82% of the time on all of them?" (There is no single answer. That is the point.)
- "Why did accuracy go up but the score go down?" (The referred cases were right more than half of the time, so half credit is worth less than answering them.)

**Typical wrong turns.**

- Students think a higher threshold makes the model better. Show both tabs side by side.
- Students set it to 99% and refer almost everything. The score sinks toward 50. Good, let them see it.
- On the adult test, 90% refers about 34% of cases and the half-credit score stays near 51.5%. Abstention cannot rescue a model that is confidently wrong.
- The Evaluate and Break it sliders are the same setting. A student who moved it earlier and forgot sees odd numbers on Results. The Results card shows the tau it used.

### 2.5 Dishonest evaluation (Repeat x5, 16 versus 180 images)

This is the quietest of the six breaks and the easiest to skip. Do not skip it, because it is the one that explains why the 99% on the Train tab cannot be trusted. It also takes under 30 seconds of compute.

**Demo in two minutes.**

1. Train tab. In the recipe card, "Check progress on" lets you choose between "Our validation set (180 images)" and "Official validation set (16 images)". Point out that the second is tiny.
2. Scroll to Repeat x5 with new seeds and click it. After about 6 seconds a table appears with five rows, seeds 1 to 5, and three columns of accuracy: 16 images, 180 images and the Kermany test.
3. Read the Lowest to highest and Spread rows. Seeds 1 to 5 gave 81.3% to 87.5% on the 16-image set (about 6 points), 98.9% to 99.4% on the 180-image set (about 0.6 points), and 79.2% to 81.3% on the test (about 2 points). The app may print the first spread as 6.2 or 6.3 points, depending on rounding. Say "about 6".
4. Ask: "If I had to choose my best model on the 16-image column, which seed would I pick?" Then ask: "Is that model really better?"

The number to say: one image on a 16-image set is worth 6.25 points. The score moves in steps of 6.25 and the steps are not information.

If you want one more fact: over 20 seeds, picking the best seed on the 16 images chose seed 1, whose honest test accuracy was 80.8%, against a 20-seed mean of 80.9%. The choice bought nothing.

**What students do.** Run Repeat x5, copy the three Spread numbers onto the handout, and write one sentence: "I would not trust a score from 16 images because ..."

**Typical wrong turns.**

- Thinking Repeat x5 replaced their model. It does not. It trains five throwaway models with seeds 1 to 5 and leaves the current one alone. It also ignores the seed box.
- Taking the 99% validation score as the honest score. The 180 images come from the same pool as training, so they are optimistic. The test set is the honest one.
- Looking for a Log this button on the Repeat table. There is none. See section 4 for how to log an "evaluation" finding.

---

## 3. Predict, then break

### Why it works

Most students, left alone, click around until something looks broken. They learn that the model can fail, which they already believed. A prediction written down first turns each click into a test. When the prediction is wrong, the student is surprised, and surprise is what makes people remember. When it is right, they now have a reason, and that reason is the hypothesis for the log.

### The card

One line, on paper, filled in before touching Break it:

> It will fail when ______ because ______.

Strong predictions are specific: "It will fail when the picture is rotated about 20 degrees because it only saw upright X-rays." Weak ones are vague: "It will fail when the image is bad." If a student writes a weak one, ask "How would I know when I had found it?" and let them rewrite.

Nobody is graded on being right. Tell them so. Being wrong is more useful.

### The 25-minute round on Day 1 (1:15 to 1:40)

| Minutes | What happens |
| --- | --- |
| 0 to 3 | Hand out the cards. Students write a prediction before opening Break it. Walk the room and push for specific ones. |
| 3 to 7 | You demo one slider (Perturb, brightness -0.40) and the webcam on a cup, once. Do not demo more. They will do the rest. |
| 7 to 20 | Free hunt in pairs. Aim for three different breaks per pair: Perturb, Your images & webcam, and the Log this buttons on the Evaluate list. Each pair logs at least three entries with a tag and a hypothesis. |
| 20 to 23 | Each pair picks its best failure and finishes the hypothesis using the template in section 4. They check the card against the entry: was the prediction right? |
| 23 to 25 | Stop. Everyone opens the Failure log tab and reads their own tag counts. Debrief follows after the break. |

If you are running late, shorten the free hunt, not the demo and not the card.

Optional for fast pairs: Repeat x5 on the Train tab. Leave the stamp lab for Day 2.

### Pairing

Pair two laptops, not one. Both students train their own model, so both have a model to break. One drives while the other writes the hypothesis, and they swap at the 13-minute mark. A beginner next to someone who is comfortable gets help without you running around.

Give the pairs different starting points so the board is not all one colour at the debrief. Tell the left half of the room to start with Perturb, the middle with the webcam and uploads, and the right half with the Evaluate list. After ten minutes they can go anywhere.

Mixed seeds are fine on Day 1. Different numbers on different laptops is itself a small lesson.

---

## 4. The Failure log

### What a good entry is

The app asks for five things, and an entry is only useful if all five are there.

| Field | Where it comes from | What can go wrong |
| --- | --- | --- |
| Image | A 96 px thumbnail of the exact pixels the model saw, including any stamp or slider change. | Nothing, it is automatic. |
| Prediction | The model's answer. "Refer to a doctor (leaning pneumonia)" when it abstained. | None. |
| Confidence | The "N% sure" and P(pneumonia). | Students ignore it. A 96% wrong answer is a different story from a 55% wrong answer. |
| Tag | Chosen by the student from seven. | Wrong tag, or "other" left on from a one-click log. |
| Hypothesis | A sentence the student types, up to 600 characters. | Missing. The tag says what, the hypothesis says why. |

Each entry also stores the settings in force: balance mode, epochs, learning rate, seed, adult mix, augmentation, refer threshold and the feature mode. That means two entries from different seeds can be compared fairly, and it is in the JSON export.

The log keeps at most 100 entries. It lives in the browser, so it survives a reload and is wiped by Reset session. Do not reset between Day 1 and the Day 2 recap.

### The seven tags

Use the app's wording as the starting point. Add the examples from this list.

**imbalance.** One class is much more common in the training data, so the model leans towards it.
- Example from this app: Evaluate, Kermany test, kermany_test_normal_0004, a healthy child called pneumonia with 96% confidence.
- Do not confuse with shift. Imbalance happens on the same kind of patient as the training set. Shift needs a different population.

**shortcut.** The model reacts to a marker or artefact, such as a stamp, a label or a corner, instead of the lungs.
- Example from this app: Break it, Stamp, a normal X-ray with the stamp on that the stamp-trained model calls pneumonia.
- Do not confuse with quality. A shortcut is a specific cue that goes with the label in the training data. Quality is a general degradation of the picture. A stamped image is not a worse picture, it is a picture with a cue.

**shift.** The image comes from a different population or machine than the training images.
- Example from this app: Evaluate, Adult test, a patient with no finding called pneumonia.
- Do not confuse with quality. Test: if the same patient were scanned again on the training machine, would the model be right? If yes, it was quality or something about the image. If the model would still be wrong because adults, hospitals and machines simply look different, it is shift.

**quality.** The picture is darker, lower in contrast, tilted or badly framed compared with the clean training images.
- Example from this app: Perturb, brightness -0.40, a normal child's X-ray that is called pneumonia with 95% confidence. Or any mistake on the Kermany stress set.
- Do not confuse with shortcut or shift. The test that works: take the unaltered original. Was it right? If so, the change in the picture caused the mistake, and that is quality.

**closed-world.** The model must answer pneumonia or normal even when the picture shows something else, such as a cup or a hand.
- Example from this app: Your images & webcam, a drawn cup called pneumonia 83% sure.
- Do not confuse with shift. A shift picture is still a chest X-ray. A closed-world picture is not one.

**evaluation.** The score itself misleads: too few test images, the wrong metric, or test data that leaked into training.
- Example from this app: Repeat x5, the 16-image validation score jumping between 81.3% and 87.5%. Or reading "82% accuracy" while specificity is 54%.
- Do not confuse with the others. Every other tag is about a picture. This one is about the number.

**other.** None of the above, or not sure yet. It is allowed and useful.
- Example from this app: any one-click log from the Stamp or webcam that was never edited, or a mistake nobody can explain.
- Rule: "other" must come with a hypothesis that says what the student is unsure about.

### Same mistake, different tag

Put this on the board, because it is the core of the whole log. A healthy child called pneumonia gets three different tags depending on where it appeared:

| Where | What you see | Tag |
| --- | --- | --- |
| Kermany test, original picture | The model leans to pneumonia | imbalance |
| Kermany stress, same picture, darker and tilted | It was right before, wrong now | quality |
| Adult test, a different person entirely | Wrong again, more often | shift |

The error looks identical. The cause is different, and the cause decides the fix: class weights, augmentation, adult X-rays.

### Hypothesis template

> I think it failed because ______. Evidence: ______. Test: ______.

A good hypothesis names a cause, points at evidence the student has already seen, and names a change that would show whether they are right.

| Quality | Example |
| --- | --- |
| Good | "I think it failed because the picture is darker than the training images. Evidence: the original is answered correctly and the darkened copy is called pneumonia with 95% confidence. Test: retrain with augmentation on and see if the 8 of 100 changed answers falls." |
| Good | "I think it flagged this healthy child because the model leans to pneumonia. Evidence: 108 of 234 healthy children were flagged and the training set is 1 to 3. Test: switch to Balanced and see whether false alarms drop below 108." |
| Weak | "The AI is bad at dark pictures." (No evidence, no test, no cause.) |
| Weak | "Because it is not trained enough." (Untestable. Epochs barely changes any of this.) |
| Weak | "It saw a weird shape." (A guess about the model's thinking with nothing to check.) |

The question to ask a student who writes something weak: "What change could you make that would prove you wrong?"

### Gaps in the app you need to know about

- **The one-click logs.** Stamp Log this and webcam Freeze & log save at once, then show an editor. Evaluate and Perturb ask for a tag first. The one-click logs default to "other" (webcam, unstamped images) or "shortcut" (a stamped image). If students do not edit, the board is full of "other".
- **A stamp entry shows the normal model's answer.** The card says "Model said" with the normal model's prediction and confidence. The stamp-trained model's probability is only stored in the exported JSON. So a "shortcut" card can show a correct "normal" answer. Tell students to put the stamp-trained model's answer in the hypothesis text, for example "the stamp model said pneumonia 99% sure".
- **There is no button for an "evaluation" finding.** Repeat x5 has no Log this. Either have students log any Evaluate row and edit the tag to evaluation with a hypothesis about the metric, or add the evaluation entries to the board yourself. I would add them yourself.
- **Logging is not limited to failures.** Log this works on a correct prediction too. Students will end up with some entries where the model was right. That is fine if the point is a control, but ask them what each entry is a control for.

### The debrief

You have ten minutes at the end of Day 1.

1. Draw seven columns on the board, one per tag.
2. Ask each pair for their best entry. They read the image id, the model's answer and confidence, and their hypothesis. You write the image id under the tag.
3. Count aloud. Typical Day 1 counts: imbalance, quality and closed-world are full. Shortcut, shift and evaluation are empty or nearly so. Say why: "Three of the seven you have not found yet. That is not an accident. Tomorrow we find them on purpose."
4. Pick one hypothesis and ask the room: "Can we test this? What would we change, and what number would move?" Push until someone says a change and a number.
5. Show the Failure log tab with the tag chips, and mention that the same counts appear on the Results card.
6. Teaser: "A model once learned which hospital an X-ray came from, not which disease. We will do that tomorrow."

Counts are per laptop. The app has no way to combine them across the room, so the board is the combined view. Each student can press Export JSON, which saves `vision-failures-YYYYMMDD-HHMM.json` with every entry, thumbnail and setting. Tell students that is their own record, and that nothing is uploaded anywhere.

On Day 2, start with the board photo and the Failure log tab, before anyone resets anything.

---

## 5. How to harvest failures from the Evaluate tab

Students waste time clicking randomly in a table of 624 rows. Give them a route. The route works for a beginner who has never seen a confusion matrix.

### Setup

Everyone trains with the default settings (seed 42). Then the lists of mistakes are the same on every laptop, so students can say "look at kermany_test_normal_0096" and have a neighbour find it. If students used other seeds, the idea still works but the ids differ.

### Steps

1. **Train, then Evaluate, Kermany test.** The scoring runs by itself. Read the baseline row first, then the model's row.
2. **Read the confusion matrix.** Two red boxes: 108 false alarms (healthy children flagged) and 2 missed (sick children not flagged). Say: "Almost all of the failures are false alarms."
3. **Open Individual predictions.** The line above the table reads "Mistakes 110 | Referred 0 | Correct 514 of 624 X-rays". That 110 is 108 plus 2.
4. **Start with Mistakes first.** This is the default view. Mistakes come first, correct answers after. Inside each group the rows follow dataset order, not confidence order. The dataset lists normal children first, so the first 108 mistake rows are healthy children flagged, and the last 2 mistake rows are the missed sick children. At 40 rows per page, that puts the two misses on page 3.
5. **Filter by true label.** The True label chips are All (624), pneumonia (390) and normal (234). Choose pneumonia and Only mistakes (2) to see the two missed sick children directly. Choose normal and Only mistakes (108) to see the false alarms. The counts on the chips follow the class filter.
6. **Use the page size.** Rows per page offers 20, 40 or 100. Use 20 on a projector so the thumbnails are big enough to see. Use 100 when students are scanning.
7. **Read the confidence column.** There is no sort by confidence, so students scan. In the paging screenshot (page 2 of the false alarms) the confidences are mostly high: 96.4%, 82.3%, 93.2%, 96.0%, 99.0%, 75.2%. Ask them to mark each mistake as confident (90% or more) or borderline (below about 65%). Confident mistakes are the dangerous ones. The refer threshold cannot catch them.
8. **Open the heatmap on three mistakes.** Press Heatmap on a row. It opens with "Evidence for pneumonia", which is the answer the model gave for a false alarm. Warm colours show where the model found evidence. In the screenshot for kermany_test_normal_0004 the warm spot sits over the middle of the chest, over the heart and the centre of the picture, and not over the lung fields. Say this carefully: the map is only 7 by 7 cells and blurry. It is a hint, not proof. Compare it with a correctly cleared normal child (Only correct, normal).
9. **Check the other two sets.** Switch Test on to Adult test and repeat steps 2 to 5. Then Kermany stress. The matrix and the list update by themselves.
   - Adult test: 97 mistakes of 200, which is 87 false alarms and 10 misses.
   - Kermany stress: 185 mistakes of 624, which is 182 false alarms and 3 misses.
   - Same direction of error on all three sets. The lean towards pneumonia is the common thread. Shift and degraded pictures make it worse.
10. **Compare stress with the original.** The stress set uses the same 624 ids as the Kermany test. In Dataset order with the normal filter, the first page is the same 20 children in both sets. Flip between the two Test on options and compare the "Model says" column row by row. A child who was cleared on Kermany test and flagged on Kermany stress is a quality failure with a built-in control. This is a clean way to earn a quality entry from the Evaluate tab. Note that in the stress list the small thumbnail is the original picture, not the darkened one. The Heatmap button and the Log this button use the degraded picture.
11. **Raise the threshold and use Only referred.** Move the slider to 90%. A fifth chip, Only referred (157), appears (it is hidden at 50%). Mistakes drops to about 45, because the model now answers 467 cases and is wrong on about 45 of them. So about 65 of the 110 mistakes were referred, but so were about 92 correct answers. Ask: "Was that a good trade?" These two figures are derived from the 90.4% accuracy on 467 answered cases, so treat them as about.
12. **Log one or two per tag.** See the table below for where each tag can come from. Ask for a hypothesis on each, using the template from section 4.

### Where to find each tag from the Evaluate tab

| Tag | Where | What to look for |
| --- | --- | --- |
| imbalance | Kermany test, normal filter, Only mistakes | Healthy child called pneumonia |
| shift | Adult test, no finding filter, Only mistakes | Adult with no finding called pneumonia |
| quality | Kermany stress, same ids as Kermany test | Correct on Kermany test, wrong on stress |
| closed-world | Test on: Your uploaded images | A cup or a landscape given a class |
| shortcut | Not from Evaluate. Break it, Stamp | Stamped normal image flipped |
| evaluation | Not from Evaluate. Train, Repeat x5 | A score that jumps with the seed |

### What to look for in the images

- **Markers and letters.** Many X-rays carry a small R or L in a corner. They appear in both classes, so they carry no signal, but students will ask. This is a good lead-in to the stamp lab.
- **Brightness.** Do the mistakes look darker or flatter than correct answers?
- **Rotation and framing.** A child tilted or cropped off-centre.
- **Children against adults.** On the adult set the body is bigger and the contrast is different. Ask for two visible differences.
- **Duplicates.** The official Kermany test set contains 6 pixel-identical duplicate pairs. If one is a mistake, the other is too, and the 110 counts the same picture twice. A duplicate is a good case of "a count can mislead".

### What not to conclude from one image

One heatmap, one dark image or one confident mistake is an anecdote. It can start a hypothesis. It cannot confirm one.

The fix is to turn the anecdote into a rate with a control. If a student says "the mistakes look dark", ask: "How many of the first 20 mistakes look dark, and how many of the first 20 correct normal children do?" If it is 12 of 20 against 3 of 20, there is something. If it is 8 of 20 against 7 of 20, there is not. Then confirm with a controlled change: the Perturb brightness slider, where brightness -0.40 changes 8 of 100 answers.

Three rules to give students:

1. Count, do not point. "Several" is not a number.
2. Always look at the correct ones too. Without a control group, any pattern is a coincidence waiting to happen.
3. A heatmap is a hint, not an explanation.

### Worked example (seed 42, Kermany test)

The app says 110 mistakes out of 624. The matrix splits them into 108 healthy children flagged (46.2% of the 234 normal children) and 2 sick children missed (0.5% of 390).

What this means:

- Most failures are false alarms. A hospital using this model would send a large number of healthy children for extra checks, which costs time and worry, but almost no sick child would go home unnoticed.
- The reason is the model's lean. The training set is 1 normal to 3 pneumonia, and the model says "pneumonia" for about 80% of everything, against 62.5% truly sick.
- Accuracy hides this. 82.4% sounds fine, but the lazy baseline gets 62.5%, and specificity is only 53.8%.
- The cost note on the Evaluate page says a miss usually costs more than a false alarm. That is why the model's lean is not obviously wrong for a screening tool. It does mean you must report both numbers.

A short good entry from this list: kermany_test_normal_0004, true normal, model pneumonia 96%, tag imbalance, hypothesis: "I think the model flags healthy children because the training set has three times more pneumonia. Evidence: 108 of 234 normal children flagged and 2 of 390 pneumonia missed. Test: Balanced class weights should reduce the 108."

Then do the same quick count on the other sets (97 on adult, 185 on stress) and say that the same direction of error is a clue, not a coincidence.

---

## 6. The Results card

Results is the one place where all three test sets, the settings and the failure counts sit on one page. It mirrors the printed handout.

### The leaderboard value

The big blue number is the worst-case score: the lowest of three half-credit scores, one per test set. Default model: **51.5%, set that set the minimum: Adult test, tau 50%, coverage 100%**.

Why the minimum. A deployed model does not get to choose which patients arrive. The number should answer "how bad can it get?" and not "how good can it look?" Taking the best or the average would let a student make one set shine and ignore the others. The minimum forces them to improve the weakest set.

The three sets, and what each one asks:

- Kermany test: does it work on patients like the training set? 624 children.
- Adult test: does it work on a different population? 200 adults, 100 pneumonia and 100 no finding, hand-balanced so the baseline is exactly 50%.
- Kermany stress: does it survive image-quality changes? The same 624 children, made darker (-0.2), lower in contrast (0.8), rotated 5 degrees and cropped to 95%. The preset is fixed so every student gets the same stress.

### Balanced accuracy, not accuracy

At tau 50% the half-credit score is balanced accuracy: the average of sensitivity and specificity. That is why the stress set contributes 60.7% even though the Evaluate tab shows accuracy 70.4%.

Why balanced. Accuracy is the share of all images called correctly, so a model that leans to the common class gets rewarded for it. Balanced accuracy gives the two classes equal weight. A model that ignores the picture and answers one thing always scores 50%, on any set. Accuracy cannot make that promise: the Kermany baseline gets 62.5% accuracy but 50% balanced.

The adult set is only 1.5 points above the floor (51.5% against 50%), which is the number to say out loud for shift.

### Half credit and the refer threshold

The score per set is the average over the two classes of (correct plus half the referred) divided by the size of the class. So a referred case counts as half a point.

- Refer everything: both classes get 0.5, the score is 50, the same as the lazy baseline.
- At tau 90% on Kermany test, accuracy on answered cases is 90.4%, but the half-credit score is 74.6%, below the 76.7% at 50%.
- In the dry run, with every fix on, tau 80% took the worst case from 59.0% to 56.0%, and 90% took it to 55.8%.

So the refer threshold buys safety, not score. Say so. It is a good place to say what the leaderboard does not measure: it measures how often the model is right, not how costly its wrong answers are.

### The buttons

- **Evaluate all three sets now.** Scores any of the three sets that were not scored yet. Greyed out when all three are done. If a student already opened each set on Evaluate, they are already scored.
- **Re-run all three with the current settings.** Redoes all three, for example after switching the precomputed feature mode.
- **Status line.** "3 of 3 sets evaluated. Features: precomputed."
- **Download results JSON.** Saves `vision-results-YYYYMMDD-HHMM.json` with the config, tau, leaderboard value, per-set scores, snapshots and failure counts.
- **Print.** Prints the card. The control buttons are hidden for printing.

### Reading the card from the top

1. The leaderboard box. Until all three sets are scored it reads "Not complete yet".
2. Scores per test set. Two rows per set, our model and the baseline, with Accuracy, Sensitivity, Specificity, Balanced accuracy, Referred n (%), Coverage, Half-credit score and Features. The set that set the minimum is tagged "(worst case)". With tau above 50%, the first three columns say "(answered cases)".
3. The model behind these numbers. Balance mode, adult X-rays mixed in, augmentation, epochs, learning rate, seed, refer threshold and "Trained on stamped images: no". This is the first place to look when a student's numbers differ from yours.
4. Failures you logged. The seven tag chips with counts, and the total. This links the Failure log to Results, and the counts are a handy summary at the end of Day 1.
5. Snapshots.

### Stale numbers: what updates and what does not

- **Changing the threshold** recomputes everything from stored predictions. A yellow note says "Recomputed for the current threshold (tau = 90%)". Nothing is stale.
- **Retraining** discards every stored evaluation. Results goes back to "Not complete yet" and the snapshot button is disabled. Press Evaluate all three sets now again.
- **Changing the recipe on Data** does nothing to a model that already exists. The card describes the model that was trained, not the one the Data tab now shows. Press Train again.
- **Switching the precomputed switch** after some sets were scored triggers a yellow note naming the sets that used the other mode. Re-run all three.

### Snapshots

A snapshot saves all three sets' scores and the model's settings under a name. Up to 8. Save is disabled until all three sets are scored. Suggested names: "Before fix", "After fix 1", "After fix 2". Every column after the first shows the change in points against the column before it, in green or red. The delta is calculated from the rounded values shown on screen, so it adds up to what students can see.

Each snapshot keeps its own tau and settings, shown in the Model row. That means deltas between two snapshots at different thresholds mix two effects. Save the fix snapshots at tau 50%, and save a separate snapshot for the threshold experiment.

### How to run the Day 2 "Fix it" round

1. **Evaluate all three** on the model students already have, check the leaderboard value reads 51.5% with seed 42 (a student who changed the seed will see something near it), and **save "Before fix" now**. After retraining the old model is gone and cannot be snapshotted.
2. In the Data tab, choose at least two fixes: Balanced, Mix in adult X-rays (+200), Augmentation (+2 views per image). Remind them the model on screen does not change until they press Train again.
3. Train again with the same seed, then Evaluate all three, then Save snapshot "After fix 1". Repeat for each further fix.
4. Try the Refer to a doctor slider, note that the worst case barely moves or falls, and save one extra snapshot for it.
5. Post the worst-case score to the leaderboard.

What students should expect, measured with seed 42, adding one fix at a time:

| Step | Kermany test | Adult test | Kermany stress | Worst case |
| --- | --- | --- | --- | --- |
| Baseline (realistic) | 76.7% | 51.5% | 60.7% | 51.5% |
| Balanced class weights | 79.9% | 51.5% | 64.9% | 51.5% |
| Plus adult X-rays mixed in | 79.2% | 59.5% | 70.1% | 59.5% |
| Plus augmentation | 77.0% | 59.0% | 78.5% | 59.0% |
| Plus refer below 80% | 74.9% | 56.0% | 76.8% | 56.0% |
| Plus refer below 90% | 74.1% | 55.8% | 73.6% | 55.8% |

Points worth making:

- The adult test is the weakest set at every step, so the leaderboard is decided by distribution shift.
- Class weights alone move nothing on the worst-case line. A student who tries only Balanced and sees 51.5% to 51.5% will think nothing happened. It did help the Kermany sets. It just is not the weakest.
- Only the adult mix lifts the worst case, by 8 points.
- Augmentation repairs the stress set (70.1% to 78.5%) but the worst case drops half a point, and the Kermany test falls by about 2 points. A fix can help one set and cost another.
- The adult mix changes how the model fails. Adult sensitivity falls from 90% to 41% while specificity rises from 13% to 78%. The score improved, but a sick adult is now missed more often than caught. Use this at the closer to ask: "Is a better score a safer model?"

### Running the leaderboard fairly

Students will not have identical numbers. Seeds differ, recipes differ, and a student who rushed may have a stale evaluation. Put these on the board, one line per student:

| Name | Worst-case score | Set that set it | Seed | Fixes on | Tau |
| --- | --- | --- | --- | --- | --- |

Rules that keep it honest:

- Post at tau 50%. A higher tau lowers the score, so a tau column prevents mixed comparisons. Keep a second, informal column for "what it scored at 90%".
- Compare scores only when the Fixes column makes sense. A student with adult mix and one without are not in the same race.
- Differences of a point or two are noise. Over 20 seeds the Kermany test accuracy spread was 3.5 points. I did not measure the spread of the adult test over seeds, so I cannot give you an exact tolerance. Treat anything inside about 2 points as a tie until you have tried a few seeds yourself.
- Everyone can check the same starting point: with seed 42 and default settings, Before fix is 51.5%. If a student's Before fix is far from that, look at "The model behind these numbers".
- All fixes are scored on the same test sets the student can look at. That is the dishonest-evaluation break in miniature: if students pick their fix by watching the test numbers, the leaderboard overstates how well they would do on new data. Say so at the end. It is a good honest note, and the Failure log has a tag for it.

### Common mistakes on Results

- Retraining before saving "Before fix". The old model and its evaluations are gone.
- Changing the recipe on Data and expecting the card to change. Press Train again.
- Reading accuracy instead of the half-credit score. The big number is the half-credit score.
- Raising the threshold to climb the leaderboard. It falls.
- Saving snapshots at different thresholds and reading the delta as the effect of a fix.
- Pressing Reset session to fix a small problem. It deletes the log and the snapshots.
- Comparing a precomputed run with a live-extraction run. They are identical to within a small amount, but the Features column says which one produced each number.

---

## 7. One-page scripts

Say these in your own words. They are meant to be short enough to remember, not read.

### Break it

Say it just before the round on Day 1, then again as a short version on Day 2.

"Your model scored 99% on validation. Hold that number in your head, because I am going to take it away. Break it is where we stop asking how high the score is and start asking how the model fails. There are six ways we will try, and you will find the first few yourselves today. Write your prediction on the card first: it will fail when this happens, because of that. Then try to prove yourself wrong. Move one slider at a time. Point the camera at a cup. Open the Evaluate list and look at the mistakes. When you see something, log it. The goal is not to find a single dramatic failure. It is to find different kinds, because a different cause needs a different fix. You will not hurt anything. Nothing leaves your laptop."

Questions to ask:

1. "What is the smallest change to a picture that makes the model change its answer?"
2. "Did the model ever say it did not know? Why not?"
3. "If a score of 99% can go to 51%, which number do you trust, and why?"

### The Failure log

Say it at the start of the hunt, and again at the debrief.

"Every failure you find, write it down. A failure without a reason is just a complaint. The log asks for five things: the image, what the model said, how sure it was, a tag for why, and a hypothesis. The tag picks one of seven causes. The hypothesis is one sentence you could test. 'I think it failed because the picture is darker, I saw it was right on the original, and I would test it by retraining with augmentation.' That is a scientist's sentence. 'The AI is bad' is not. Be careful with the tag. A healthy child called pneumonia can be imbalance, shift or quality, depending on what you changed to get there. The tag follows the cause, not the look of the mistake. If you do not know, write other and say what you are unsure about. By the end I want one entry for every tag you can find."

Questions to ask:

1. "This image was right yesterday and wrong today. What changed, and which tag is that?"
2. "Which of your hypotheses could you test in the next five minutes?"
3. "Which tags are empty on the board? What would we have to do to fill them?"

### Results

Say it at the start of the Fix it round on Day 2.

"This is your report card. The big number is not your best result. It is your worst one, taken across three test sets: children from the same hospital, adults from another hospital, and the same children with worse pictures. A model is only as good as its worst set. At the default, the worst set is the adults, at about 51.5%, and the lazy rule 'always say pneumonia' already gets 50. The score is balanced accuracy, so it cannot be gamed by leaning towards the common class. If the model hands a case to a doctor, it gets half a point, so handing everything over scores exactly 50. That means the Refer slider buys you safety and not a higher score. Save a snapshot called Before fix now, because once you retrain, the old model is gone. Then apply two fixes, retrain, press Evaluate all three, save a second snapshot and look at the change. Post your worst-case score with the fixes you used."

Questions to ask:

1. "Which of the three sets is your weakest, and what does that tell you to fix first?"
2. "Your score went up. Did the model get safer? Look at adult sensitivity before and after."
3. "Why did accuracy go up when you raised the threshold, but the score went down?"

---

## 8. Things in the app to know before you teach

I found these while reading the code and screenshots. They are not blockers, but each could surprise you in front of the room.

1. **The UI guide is out of date on the prediction list.** `docs/ui-guide.md` section 2.3 still says "the first 40 rows, Show mistakes first ticked by default". The app now has Show chips (Mistakes first, Only mistakes, Only referred, Only correct, Dataset order), True label chips, page sizes of 20, 40 and 100 and First, Previous, Next and Last buttons.
2. **Mistakes first is not sorted by confidence.** Mistakes come first in dataset order. You cannot sort by "most confident mistake", so finding the worst confident errors takes scanning.
3. **The prediction list thumbnail on Kermany stress is the original picture.** The Heatmap and Log this use the degraded picture. This can confuse students who expect a dark image in the row.
4. **A stamp log entry shows the normal model's answer.** The stamp-trained model's probability is stored only in the JSON.
5. **There is no way to log an "evaluation" finding** from the Repeat x5 card.
6. **There is no link between an Evaluate row and the Perturb preview.** Perturb uses a fixed 100-image sample and its picker lists the first 40 by position. A student cannot take kermany_test_normal_0096 into the sliders.
7. **The Heatmap opens on "Evidence for pneumonia"** even when the model answered normal, which is an easy thing to misread for a false negative. The student must switch to "normal".
8. **The Repeat x5 spread may print 6.2 points, not 6.3.** The UI guide and the dry run say 6.3. The code subtracts the rounded values (87.5 minus 81.3), which gives 6.2. Say "about 6".
9. **Augmentation lowers the worst-case score slightly** (59.5% to 59.0%) while raising the stress set a lot. Not a bug, but students who expect each fix to increase the leaderboard number will be confused.
10. **One-click logs default to "other".** Stamp Log this and webcam Freeze & log save at once. Without the editor step, the log fills with "other".
11. **Log this also saves correct predictions.** The Log the changed image button in Perturb saves even when the answer did not change.
12. **I could not verify two things.** The flip rate after retraining with Both classes, and the seed-to-seed spread of the adult worst-case score. Neither is in the dry run. Try them once before the workshop, or avoid quoting a number.
13. **The workshop checklist** puts "Evaluate on the adult X-rays" on Day 1 and "Test on the adult set and note the drop" on Day 2. Present the Day 1 look as a teaser, as the UI guide says.
