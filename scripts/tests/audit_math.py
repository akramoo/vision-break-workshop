"""Independent re-derivation of every number the app computes from labels + probabilities.

Run (from the repository root):
    cd app && npx vitest run tests/unit/mathAudit.test.ts          # writes app_outputs.json
    .venv/Scripts/python scripts/tests/audit_math.py      # this script

Both use $MATH_AUDIT_OUT (default: <tmp>/vision-math-audit). Exit code 1 on any mismatch.
Nothing here imports app code: labels come from manifest.json, features from the .f16.bin files (numpy float16).
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "app" / "public" / "data"
OUT = Path(os.environ.get("MATH_AUDIT_OUT", Path(tempfile.gettempdir()) / "vision-math-audit"))
TAUS = [0.5, 0.7, 0.8, 0.9]
FAIL = []


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        FAIL.append(msg)


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


manifest = json.loads((DATA / "manifest.json").read_text())
by_id = {e["id"]: e for e in manifest}
app = json.loads((OUT / "app_outputs.json").read_text())


def ids_of(source, split):
    return [e["id"] for e in manifest if e["source"] == source and e["split"] == split]


def label(i):
    return 1 if by_id[i]["label"] == "pneumonia" else 0


# ---------- embeddings: float16 decode and id lookup ----------
def load(variant):
    ix = json.loads((DATA / "embeddings" / f"{variant}.index.json").read_text())
    x = np.frombuffer((DATA / "embeddings" / f"{variant}.f16.bin").read_bytes(), dtype="<f2").reshape(ix["rows"], 1280)
    keys = [f"{i}|{v}" for i, v in zip(ix["ids"], ix["views"])] if variant == "aug" else ix["ids"]
    return ix, x, {k: r for r, k in enumerate(keys)}


base_ix, base, base_row = load("base")
js_dec = np.fromfile(OUT / "base_decoded.f32", dtype="<f4").reshape(-1, 1280)
check(np.array_equal(base.astype(np.float32), js_dec, equal_nan=True), "float16 decode: JS table decode == numpy float16 (bit-exact, 2240 x 1280)")
stress_ix, stress, stress_row = load("stress")
test_ids = ids_of("kermany", "test")
check(sorted(stress_ix["ids"]) == sorted(test_ids), "stress rows are exactly the 624 Kermany test ids")
check(stress_ix["ids"] == test_ids, "stress index is in manifest order (irrelevant: the app looks rows up by id)")


def feats(variant_rows, mat, ids):
    return mat[[variant_rows[i] for i in ids]].astype(np.float64)


# ---------- leakage ----------
splits = {(e["source"], e["split"]) for e in manifest}
sets = {s: set(ids_of(*s)) for s in splits}
train = sets[("kermany", "train")] | sets[("nih", "train_adult")]
held = sets[("kermany", "val")] | sets[("kermany", "val_official")] | sets[("kermany", "test")] | sets[("nih", "test_adult")]
check(not (train & held), "no id is in both a training split and a val/test split")
check(set(app["plan"]["baseIds"]) == sets[("kermany", "train")] and not (set(app["plan"]["baseIds"]) & held), "training ids = Kermany train only (default recipe)")
digest = {}
for e in manifest:
    digest.setdefault(hashlib.sha256((DATA / e["path"]).read_bytes()).hexdigest(), []).append(e)
dups = [v for v in digest.values() if len({(x["source"], x["split"]) for x in v}) > 1]
cross = [v for v in dups if any((x["source"], x["split"]) in {("kermany", "train"), ("nih", "train_adult")} for x in v)
         and any((x["id"] in held) for x in v)]
check(not cross, f"no byte-identical JPEG shared between training and held-out splits ({len(cross)} found)")
# exact duplicate feature rows between train and held-out
tr_set = {r.tobytes() for r in base[[base_row[i] for i in sorted(train)]]}
emb_dup = sum(r.tobytes() in tr_set for r in base[[base_row[i] for i in sorted(held)]])
check(emb_dup == 0, f"no held-out embedding row identical to a training row ({emb_dup})")

# ---------- plans / class weights ----------
P = app["plans"]
exp = {
    "realistic": (1020, 255, 765, None),
    "balanced": (1020, 255, 765, (1020 / 510, 1020 / 1530)),
    "balanced_adult": (1220, 355, 865, (1220 / 710, 1220 / 1730)),
    "balanced_adult_aug": (3660, 1065, 2595, (3660 / 2130, 3660 / 5190)),
    "realistic_aug": (3060, 765, 2295, None),
}
for k, (n, n0, n1, w) in exp.items():
    p = P[k]
    ok = (p["nRows"], p["nNormal"], p["nPneumonia"]) == (n, n0, n1) and p["rowLabels"] == n
    if w is None:
        ok &= p["weights"] is None
    else:
        ok &= abs(p["weights"]["0"] - w[0]) < 1e-12 and abs(p["weights"]["1"] - w[1]) < 1e-12
    check(ok, f"plan {k}: rows {p['nRows']} = {p['nNormal']} normal + {p['nPneumonia']} pneumonia, weights {p['weights'] or p['balancedWeights']}")
print("     balanced weights shown on the Data tab: realistic 2.00/0.67, +adult {:.2f}/{:.2f}".format(1220 / 710, 1220 / 1730))


# ---------- metrics, independent ----------
def metrics(y, p, tau):
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    conf = np.maximum(p, 1 - p)
    refer = conf < tau
    call = p >= 0.5
    ans = ~refer
    tp = int(np.sum(ans & (y == 1) & call)); fn = int(np.sum(ans & (y == 1) & ~call))
    fp = int(np.sum(ans & (y == 0) & call)); tn = int(np.sum(ans & (y == 0) & ~call))
    r = lambda a, b: a / b if b > 0 else None
    sens, spec = r(tp, tp + fn), r(tn, tn + fp)
    n = len(y)
    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    ref0, ref1 = int(np.sum(refer & (y == 0))), int(np.sum(refer & (y == 1)))
    c0 = r(tn + 0.5 * ref0, n0); c1 = r(tp + 0.5 * ref1, n1)
    return dict(confusion=dict(tp=tp, fn=fn, fp=fp, tn=tn), accuracy=r(tp + tn, tp + tn + fp + fn), sensitivity=sens,
                specificity=spec, balancedAccuracy=None if sens is None or spec is None else (sens + spec) / 2,
                n=n, referred=int(refer.sum()), referredFraction=r(int(refer.sum()), n), coverage=(n - int(refer.sum())) / n,
                classCredit=dict(normal=c0, pneumonia=c1), referredByClass=dict(normal=ref0, pneumonia=ref1),
                halfCreditScore=None if c0 is None or c1 is None else (c0 + c1) / 2)


def same(a, b):
    if isinstance(a, dict):
        return all(same(a[k], b[k]) for k in a)
    return a == b  # exact, including floats


labels_by_set = {}
for sid, s in app["sets"].items():
    src, split = {"kermany_test": ("kermany", "test"), "adult_test": ("nih", "test_adult"), "stress": ("kermany", "test")}[sid]
    check(s["ids"] == ids_of(src, split), f"{sid}: ids are the manifest ids of {src}/{split}")
    y = [label(i) for i in s["ids"]]
    labels_by_set[sid] = y
    check(y == s["labels"], f"{sid}: labels match the manifest")
    for tau in TAUS:
        mine = metrics(y, s["probs"], tau)
        check(same(mine, s["metrics"][str(tau)]), f"{sid} tau {tau}: app metrics == independent (exact) "
              f"acc {pct(mine['accuracy'])} sens {pct(mine['sensitivity'])} spec {pct(mine['specificity'])} "
              f"bal {pct(mine['balancedAccuracy'])} ref {mine['referred']} half {pct(mine['halfCreditScore'])}")
    b = s["baseline"]
    yy = np.asarray(y)
    check(b["accuracy"] == yy.mean() and b["sensitivity"] == 1 and b["specificity"] == 0 and b["balancedAccuracy"] == 0.5,
          f"{sid}: baseline {pct(b['accuracy'])}/100/0")
    # an all-refer model scores exactly 0.5
    check(metrics(y, [0.5] * len(y), 0.51)["halfCreditScore"] == 0.5, f"{sid}: refer everything -> half credit 0.5")

for tau in TAUS:
    sm = app["summaries"][str(tau)]
    hc = {sid: metrics(labels_by_set[sid], app["sets"][sid]["probs"], tau)["halfCreditScore"] for sid in ["kermany_test", "adult_test", "stress"]}
    worst = min(hc, key=lambda k: hc[k])
    check(sm["leaderboard"]["score"] == hc[worst] and sm["leaderboard"]["set"] == worst,
          f"leaderboard tau {tau}: {pct(hc[worst])} on {worst} (app {pct(sm['leaderboard']['score'])} {sm['leaderboard']['set']})")
    check(all(sm["perSet"][k]["baseline"]["halfCreditScore"] == 0.5 for k in hc), f"tau {tau}: baseline half credit 50 on every set")

# ---------- probabilities from the saved weights (numpy float64) ----------
w = np.asarray(app["weights"]["kernel"], dtype=np.float64)
b = app["weights"]["bias"]
sig = lambda z: 1 / (1 + np.exp(-z))
for sid, var in [("kermany_test", None), ("adult_test", None), ("stress", "stress")]:
    rows, mat = (stress_row, stress) if var else (base_row, base)
    p64 = sig(feats(rows, mat, app["sets"][sid]["ids"]) @ w + b)
    pa = np.asarray(app["sets"][sid]["probs"])
    d = np.abs(p64 - pa).max()
    flips = int(np.sum((p64 >= 0.5) != (pa >= 0.5)))
    check(d < 1e-5 and flips == 0, f"{sid}: predictProba == numpy sigmoid(x.w+b) (max diff {d:.2e}, decision flips {flips})")

# ---------- training history (unweighted BCE, accuracy at p >= 0.5, full sets) ----------
h = app["history"][-1]
Xtr = feats(base_row, base, app["plan"]["baseIds"]); ytr = np.array([label(i) for i in app["plan"]["baseIds"]])
Xv = feats(base_row, base, app["plan"]["valIds"]); yv = np.array([label(i) for i in app["plan"]["valIds"]])


def loss_acc(X, y):
    z = X @ w + b
    l = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    return l.mean(), ((z >= 0) == (y == 1)).mean()


tl, ta = loss_acc(Xtr, ytr); vl, va = loss_acc(Xv, yv)
check(abs(h["trainLoss"] - tl) < 1e-4 and abs(h["valLoss"] - vl) < 1e-4 and abs(h["trainAcc"] - ta) < 1e-6 and abs(h["valAcc"] - va) < 1e-6,
      f"epoch-20 stats: train loss {tl:.4f} acc {pct(ta)}, val loss {vl:.4f} acc {pct(va)} (app {h['trainLoss']:.4f}/{pct(h['trainAcc'])}/{h['valLoss']:.4f}/{pct(h['valAcc'])})")

# ---------- reported default numbers ----------
k = app["sets"]["kermany_test"]["metrics"]["0.5"]; a = app["sets"]["adult_test"]["metrics"]["0.5"]; s = app["sets"]["stress"]["metrics"]["0.5"]
check(pct(h["valAcc"]) == "98.9%", f"validation accuracy {pct(h['valAcc'])} (docs: 98.9%)")
check((pct(k["accuracy"]), pct(k["sensitivity"]), pct(k["specificity"])) == ("82.4%", "99.5%", "53.8%"),
      f"Kermany test {pct(k['accuracy'])}/{pct(k['sensitivity'])}/{pct(k['specificity'])} bal {pct(k['balancedAccuracy'])} cm {k['confusion']} (docs 82.4/99.5/53.8, 126/108/2/388)")
# adult and stress each have normals with p within 0.0005 of 0.5, so the tf backend (WebGL in the browser,
# pure-JS CPU here) can move one image: accept the documented value +- one image and say which one we got.
check(abs(a["accuracy"] - 0.515) <= 1 / 200 + 1e-12, f"adult test {pct(a['accuracy'])}/{pct(a['sensitivity'])}/{pct(a['specificity'])} cm {a['confusion']} (docs, WebGL: 51.5/90/13; +-1 image allowed)")
check(abs(s["accuracy"] - 439 / 624) <= 1 / 624 + 1e-12, f"stress {pct(s['accuracy'])}/{pct(s['sensitivity'])}/{pct(s['specificity'])} bal {pct(s['balancedAccuracy'])} cm {s['confusion']} (docs, WebGL: 70.4/99.2/22.2; +-1 image allowed)")
for sid in ["adult_test", "stress"]:
    pp = np.asarray(app["sets"][sid]["probs"])
    print(f"     {sid}: {int(np.sum(np.abs(pp - 0.5) < 1e-3))} image(s) with |p - 0.5| < 0.001 (backend-sensitive)")
fl = k["confusion"]["fp"] / 234; said = (k["confusion"]["fp"] + k["confusion"]["tp"]) / 624
print(f"     Kermany: flags {pct(fl)} of healthy children, says pneumonia for {pct(said)}")
lb = app["summaries"]["0.5"]["leaderboard"]
print(f"     worst case at tau 0.5: {pct(lb['score'])} ({lb['set']})")
for tau in TAUS:
    m = app["sets"]["kermany_test"]["metrics"][str(tau)]
    print(f"     Kermany tau {tau}: answered-acc {pct(m['accuracy'])}, referred {m['referred']} ({pct(m['referredFraction'])}), half {pct(m['halfCreditScore'])}")

# ---------- weighted vs unweighted vs sklearn ----------
from sklearn.linear_model import LogisticRegression

bal = app["balanced"]["sets"]
for sid in ["kermany_test", "adult_test", "stress"]:
    y = labels_by_set[sid]
    r_, b_ = metrics(y, app["sets"][sid]["probs"], 0.5), metrics(y, bal[sid]["probs"], 0.5)
    print(f"     app {sid}: realistic sens/spec {pct(r_['sensitivity'])}/{pct(r_['specificity'])}  balanced {pct(b_['sensitivity'])}/{pct(b_['specificity'])}")
kb = metrics(labels_by_set["kermany_test"], bal["kermany_test"]["probs"], 0.5)
check(kb["specificity"] > k["specificity"] and kb["sensitivity"] <= k["sensitivity"], "app: class weights raise Kermany specificity and do not raise sensitivity")
for name, cw in [("realistic", None), ("balanced", "balanced")]:
    clf = LogisticRegression(C=1.0, class_weight=cw, max_iter=5000).fit(Xtr, ytr)
    row = []
    for sid, (rows, mat) in [("kermany_test", (base_row, base)), ("adult_test", (base_row, base)), ("stress", (stress_row, stress))]:
        pp = clf.predict_proba(feats(rows, mat, app["sets"][sid]["ids"]))[:, 1]
        m = metrics(labels_by_set[sid], pp, 0.5)
        row.append(f"{sid} acc {pct(m['accuracy'])} sens {pct(m['sensitivity'])} spec {pct(m['specificity'])}")
        if sid == "kermany_test":
            ref = k if cw is None else kb
            check(abs(m["accuracy"] - ref["accuracy"]) < 0.05, f"sklearn {name} Kermany acc {pct(m['accuracy'])} within 5 points of app {pct(ref['accuracy'])}")
    print("     sklearn " + name + ": " + "; ".join(row))

# ---------- perturb samples ----------
for key, v in app["perturb"].items():
    sid, size = key.rsplit("_", 1); size = int(size)
    y = np.asarray(labels_by_set[sid]); idx = v["idx"]
    exp_pos = int(np.floor(size * y.sum() / len(y) + 0.5))
    check(len(idx) == size and len(set(idx)) == size and idx == sorted(idx) and v["nPos"] == exp_pos == int(y[idx].sum())
          and v["baseline"]["accuracy"] == exp_pos / size,
          f"perturb sample {key}: {size} images, {exp_pos} pneumonia, baseline {pct(exp_pos / size)}")

# ---------- stamp lab ----------
st = app["stamp"]; y = np.asarray(st["labels"])
for which in ["before", "after"]:
    c = np.asarray(st[which]["cleanProbs"]) >= 0.5; sp = np.asarray(st[which]["stampedProbs"]) >= 0.5
    nm = y == 0
    flipped = int(np.sum(c[nm] != sp[nm]))
    sm = st[which]["summary"]["normal"]
    check(sm["flip"]["flipped"] == flipped and sm["n"] == 234 and sm["pneumoniaClean"] == c[nm].mean() and sm["pneumoniaStamped"] == sp[nm].mean(),
          f"stamp {which}: normal flip {flipped}/234 = {pct(flipped / 234)}, called pneumonia clean {pct(c[nm].mean())} -> stamped {pct(sp[nm].mean())}")
    pm = st[which]["summary"]["pneumonia"]
    check(pm["flip"]["flipped"] == 0 and pm["pneumoniaClean"] == pm["pneumoniaStamped"], f"stamp {which}: pneumonia test images unstamped -> 0 flips")
    cm = metrics(y, st[which]["cleanProbs"], 0.5)
    print(f"     stamp {which}: clean acc {pct(cm['accuracy'])} sens {pct(cm['sensitivity'])} spec {pct(cm['specificity'])}")

# ---------- repeat x5 ----------
rep = app["repeat"]
for key in ["val16", "val180", "test"]:
    vals = [r[key] for r in rep]
    rr = app["repeatRange"][key]
    check(rr["min"] == min(vals) and rr["max"] == max(vals) and rr["spread"] == max(vals) - min(vals),
          f"repeat {key}: {', '.join(pct(v) for v in vals)}; spread {(max(vals) - min(vals)) * 100:.1f} points")
check(all(pct(r["val180"]) == pct(r["historyValAcc"]) for r in rep), "repeat: JS val180 accuracy == tf epoch valAcc (as displayed)")
check(all(abs(r["val16"] * 16 - round(r["val16"] * 16)) < 1e-12 for r in rep), "repeat: val16 accuracies are multiples of 1/16 (6.25 points)")

print()
print("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED")
sys.exit(1 if FAIL else 0)
