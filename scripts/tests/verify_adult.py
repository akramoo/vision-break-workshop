"""Independent verification of D2 (adult / NIH ChestMNIST-224 images + manifest merge).
Run: .venv/Scripts/python scripts/tests/verify_adult.py [--skip-source] [--skip-rerun]
  --skip-source  skip the (slow, ~1-2 min) streaming label/split verification against the npz
  --skip-rerun   skip the idempotence check (re-runs scripts/prepare_adult.py, ~30 s)
Exit code is non-zero if any check fails.

Source verification does NOT import prepare_adult. Method: stream every row of
{train,val,test}_images.npy from the deflate npz, compute a 28x28 block-mean signature per row,
and keep (per adult JPEG) the nearest source row overall (all three ChestMNIST splits at once).
The winning row's full 224x224 pixels are then compared with the decoded JPEG (mean abs diff,
must be JPEG-noise sized) and the winner's split / labels are checked against the manifest.
Finally the sampling is recomputed from scratch with default_rng(42) as a cross-check."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from numpy.lib import format as npy_format
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "app" / "public" / "data"
MANIFEST = DATA / "manifest.json"
NPZ = ROOT / "scripts" / ".cache" / "medmnist" / "chestmnist_224.npz"
PY = ROOT / ".venv" / "Scripts" / "python.exe"
SKIP_SOURCE = "--skip-source" in sys.argv
SKIP_RERUN = "--skip-rerun" in sys.argv

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail or ""))
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""), flush=True)
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"FAIL  {name}: {e}", flush=True)
    except Exception as e:  # noqa
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name}: {type(e).__name__}: {e}", flush=True)


raw = MANIFEST.read_bytes()
manifest_sha = hashlib.sha256(raw).hexdigest()
print("manifest.json sha256:", manifest_sha, "bytes:", len(raw))
all_entries = json.loads(raw)
adult = [e for e in all_entries if e.get("source") == "nih"]
kerm = [e for e in all_entries if e.get("source") == "kermany"]
SPLITS = ("train_adult", "test_adult")
LABELS = ("normal", "pneumonia")


def adult_bytes_hash():
    h = hashlib.sha256()
    for p in sorted((DATA / "adult").rglob("*")):
        if p.is_file():
            h.update(p.relative_to(DATA).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def t_totals():
    assert len(all_entries) == 2240, f"total {len(all_entries)}"
    assert len(kerm) == 1840, f"kermany {len(kerm)}"
    assert len(adult) == 400, f"nih {len(adult)}"
    other = [e for e in all_entries if e.get("source") not in ("kermany", "nih")]
    assert not other, f"unknown sources {other[:2]}"
    bad = [e for e in all_entries if set(e) != {"id", "path", "label", "split", "source"}]
    assert not bad, f"bad keys {bad[:1]}"
    return "2240 = 1840 kermany + 400 nih"


def t_counts():
    c = Counter((e["split"], e["label"]) for e in adult)
    exp = {(s, l): 100 for s in SPLITS for l in LABELS}
    assert dict(c) == exp, f"got {dict(c)}"
    return str(dict(sorted(c.items())))


def t_ids():
    ids = [e["id"] for e in all_entries]
    assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate ids across whole manifest"
    paths = [e["path"] for e in all_entries]
    assert len(paths) == len(set(paths)), "duplicate paths across whole manifest"
    return "ids and paths unique across all 2240"


def t_paths():
    for e in adult:
        p = e["path"]
        assert "\\" not in p, f"backslash {p}"
        assert not p.startswith("/") and ":" not in p and ".." not in p.split("/"), f"not relative {p}"
        parts = p.split("/")
        assert len(parts) == 4 and parts[0] == "adult", f"layout {p}"
        assert parts[1] == e["split"] and parts[2] == e["label"], f"folder mismatch {e}"
        assert parts[3] == e["id"] + ".jpg", f"filename != id {e}"
        assert (DATA / p).is_file(), f"missing {p}"


def t_strays():
    on_disk = {p.relative_to(DATA).as_posix() for p in (DATA / "adult").rglob("*") if p.is_file()}
    listed = {e["path"] for e in adult}
    assert on_disk == listed, f"stray={sorted(on_disk - listed)[:3]} missing={sorted(listed - on_disk)[:3]}"
    dirs = {p.relative_to(DATA / "adult").as_posix() for p in (DATA / "adult").rglob("*") if p.is_dir()}
    exp = {s for s in SPLITS} | {f"{s}/{l}" for s in SPLITS for l in LABELS}
    assert dirs == exp, f"unexpected dirs {dirs ^ exp}"
    return f"{len(on_disk)} files == {len(listed)} entries, no other files/dirs"


def q85_tables():
    buf = io.BytesIO()
    arr = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, "JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).quantization


Q85 = q85_tables()
decoded = {}  # id -> uint8 [224,224] gray (R channel)


def t_images():
    bad_q = []
    for e in adult:
        p = DATA / e["path"]
        with Image.open(p) as im:
            assert im.format == "JPEG", f"{e['path']}: format {im.format}"
            assert im.mode == "RGB", f"{e['path']}: mode {im.mode}"
            assert im.size == (224, 224), f"{e['path']}: size {im.size}"
            q = im.quantization
            if q[0] != Q85[0] or q.get(1) != Q85.get(1):
                bad_q.append((e["path"], q[0][:3]))
            im.load()
            a = np.asarray(im)
        assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all(), f"{e['path']}: R,G,B differ"
        decoded[e["id"]] = np.ascontiguousarray(a[..., 0])
    assert not bad_q, f"{len(bad_q)} non-q85 files e.g. {bad_q[:2]}"
    return "400/400 JPEG RGB 224x224, q-tables == PIL q85, R==G==B everywhere"


def t_gray_all():
    return None  # covered in t_images; kept so the report reads naturally


def t_leakage():
    assert decoded, "images check must run first"
    h = {}
    for e in adult:
        h.setdefault(hashlib.sha256(decoded[e["id"]].tobytes()).hexdigest(), []).append(e)
    dup_cross = [v for v in h.values() if len({x["split"] for x in v}) > 1]
    assert not dup_cross, f"{len(dup_cross)} pixel-identical images across train_adult/test_adult"
    dup_any = [v for v in h.values() if len(v) > 1]
    assert not dup_any, f"{len(dup_any)} intra-adult pixel duplicates: {[x['id'] for x in dup_any[0]]}"
    # vs Kermany: compare decoded pixels
    kh = set()
    for e in kerm:
        with Image.open(DATA / e["path"]) as im:
            kh.add(hashlib.sha256(np.ascontiguousarray(np.asarray(im.convert("RGB"))[..., 0]).tobytes()).hexdigest())
    inter = set(h) & kh
    assert not inter, f"{len(inter)} adult images pixel-identical to a Kermany image"
    # Same JPEG file bytes too
    ab = {hashlib.sha256((DATA / e["path"]).read_bytes()).hexdigest() for e in adult}
    kb = {hashlib.sha256((DATA / e["path"]).read_bytes()).hexdigest() for e in kerm}
    assert not (ab & kb), "byte-identical file between adult and kermany"
    return "no pixel-identical pairs: train_adult vs test_adult, within adult, adult vs Kermany"


def t_gray_names():
    # Ids follow the documented scheme and are contiguous 0001..0100 per (split,label)
    for s in SPLITS:
        for l in LABELS:
            ids = sorted(e["id"] for e in adult if e["split"] == s and e["label"] == l)
            exp = [f"nih_{s}_{l}_{k:04d}" for k in range(1, 101)]
            assert ids == exp, f"{s}/{l}: ids not nih_<split>_<label>_0001..0100"
    return "ids nih_<split>_<label>_0001..0100"


# ------------------------------- source verification -------------------------------
SRC = {}  # filled by source_scan


def stream_rows(npz, key, chunk=256):
    with npz.open(f"{key}.npy") as f:
        ver = npy_format.read_magic(f)
        hdr = npy_format.read_array_header_1_0 if ver == (1, 0) else npy_format.read_array_header_2_0
        shape, fortran, dtype = hdr(f)
        assert dtype == np.uint8 and not fortran and shape[1:] == (224, 224), (key, shape, dtype)
        rb = 224 * 224
        n = shape[0]
        i = 0
        while i < n:
            k = min(chunk, n - i)
            buf = f.read(k * rb)
            assert len(buf) == k * rb, f"{key} truncated at {i}"
            yield i, np.frombuffer(buf, dtype=np.uint8).reshape(k, 224, 224)
            i += k


def sig(a):  # a: [k,224,224] -> [k,784] float32 8x8 block means
    return a.reshape(-1, 28, 8, 28, 8).mean(axis=(2, 4), dtype=np.float32).reshape(-1, 784)


def source_scan():
    ids = [e["id"] for e in adult]
    X = np.stack([decoded[i] for i in ids])  # [400,224,224]
    S = sig(X)  # [400,784]
    S2 = (S * S).sum(1)
    best = {}  # split -> (dist[400], row[400], rowdata[400,224,224])
    SRC["n"] = {}
    with zipfile.ZipFile(NPZ) as npz:
        for sp in ("test", "val", "train"):
            bd = np.full(400, np.inf, dtype=np.float64)
            br = np.full(400, -1, dtype=np.int64)
            bdata = np.zeros((400, 224, 224), dtype=np.uint8)
            n = 0
            for i0, rows in stream_rows(npz, f"{sp}_images"):
                R = sig(rows)  # [k,784]
                d = (R * R).sum(1)[:, None] + S2[None, :] - 2.0 * (R @ S.T)  # [k,400] squared dist
                am = d.argmin(0)
                val = d[am, np.arange(400)]
                upd = val < bd
                for j in np.flatnonzero(upd):
                    bd[j] = val[j]
                    br[j] = i0 + am[j]
                    bdata[j] = rows[am[j]]
                n = i0 + len(rows)
            SRC["n"][sp] = n
            best[sp] = (bd, br, bdata)
            print(f"   scanned {sp}_images: {n} rows", flush=True)
    SRC["best"] = best
    SRC["ids"] = ids
    SRC["X"] = X


def t_source_match():
    ids, X, best = SRC["ids"], SRC["X"], SRC["best"]
    splits = ("test", "val", "train")
    D = np.stack([best[s][0] for s in splits])  # [3,400]
    win = D.argmin(0)
    want = np.array([0 if e["split"] == "test_adult" else 2 for e in adult])
    wrong = np.flatnonzero(win != want)
    assert len(wrong) == 0, f"{len(wrong)} images whose nearest source row is in the wrong ChestMNIST split, e.g. {[ids[j] for j in wrong[:3]]} -> {[splits[win[j]] for j in wrong[:3]]}"
    mad = np.empty(400)
    gap = np.empty(400)
    SRC["row"] = np.empty(400, dtype=np.int64)
    for j in range(400):
        s = splits[win[j]]
        mad[j] = np.abs(best[s][2][j].astype(np.int16) - X[j].astype(np.int16)).mean()
        others = [D[k, j] for k in range(3) if k != win[j]]
        gap[j] = min(others) / max(D[win[j], j], 1e-9)
        SRC["row"][j] = best[s][1][j]
    SRC["mad"] = mad
    assert mad.max() < 3.0, f"max MAD vs matched source row {mad.max():.2f} (JPEG q85 noise should be ~1)"
    # exact-duplicate rows would make the match ambiguous: the winning row must also be unique within the manifest
    keys = {(int(win[j]), int(SRC["row"][j])) for j in range(400)}
    assert len(keys) == 400, "two manifest images matched the same source row"
    return f"400/400 matched to a unique source row; MAD mean {mad.mean():.2f} max {mad.max():.2f} (0-255 scale)"


def t_source_labels():
    import zipfile as zf
    labels = {}
    with zf.ZipFile(NPZ) as npz:
        for s in ("train", "val", "test"):
            with npz.open(f"{s}_labels.npy") as f:
                labels[s] = npy_format.read_array(f)
    SRC["labels"] = labels
    pneu_bad, norm_bad = [], []
    for j, e in enumerate(adult):
        sp = "test" if e["split"] == "test_adult" else "train"
        row = labels[sp][SRC["row"][j]]
        if e["label"] == "pneumonia":
            if row[6] != 1:
                pneu_bad.append(e["id"])
        else:
            if row.sum() != 0:
                norm_bad.append(e["id"])
    assert not pneu_bad, f"{len(pneu_bad)} 'pneumonia' images with label[6]!=1, e.g. {pneu_bad[:3]}"
    assert not norm_bad, f"{len(norm_bad)} 'normal' images with a non-zero label, e.g. {norm_bad[:3]}"
    npn = sum(1 for j, e in enumerate(adult) if e["label"] == "pneumonia"
              and labels["test" if e["split"] == "test_adult" else "train"][SRC["row"][j]].sum() > 1)
    return f"(a) 200/200 pneumonia have label[6]==1 ({npn} also have other findings), 200/200 normal have all 14 labels 0; (b) 200 test_adult in test, 200 train_adult in train"


def t_resample():
    labels = SRC["labels"]
    rng = np.random.default_rng(42)
    exp = {}
    for split, src in (("test_adult", "test"), ("train_adult", "train")):
        a = labels[src]
        pos = np.flatnonzero(a[:, 6] == 1)
        neg = np.flatnonzero(a.sum(axis=1) == 0)
        assert len(pos) >= 100, f"{src}: only {len(pos)} positives (script would hit the val fallback)"
        exp[(split, "pneumonia")] = np.sort(rng.choice(pos, size=100, replace=False))
        exp[(split, "normal")] = np.sort(rng.choice(neg, size=100, replace=False))
    for (split, label), rows in exp.items():
        got = np.array([SRC["row"][j] for j, e in enumerate(adult) if e["split"] == split and e["label"] == label])
        got_sorted_by_id = np.array([SRC["row"][j] for j, e in sorted(
            ((j, e) for j, e in enumerate(adult) if e["split"] == split and e["label"] == label), key=lambda t: t[1]["id"])])
        assert (np.sort(got) == rows).all(), f"{split}/{label}: matched rows != default_rng(42) draw"
        assert (got_sorted_by_id == rows).all(), f"{split}/{label}: ids are not in ascending source-row order"
    return "matched source rows == independent default_rng(42) draw (order test-pos,test-neg,train-pos,train-neg), ids ascend with row"


def t_md5():
    import medmnist
    exp = medmnist.INFO["chestmnist"]["MD5_224"]
    h = hashlib.md5()
    with open(NPZ, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    assert h.hexdigest() == exp, f"md5 {h.hexdigest()} != {exp}"
    return f"npz md5 {h.hexdigest()} == INFO['chestmnist']['MD5_224']"


def t_label_name():
    import medmnist
    lab = medmnist.INFO["chestmnist"]["label"]
    assert len(lab) == 14, len(lab)
    assert lab["6"].lower() == "pneumonia", f"label 6 is {lab['6']!r}"
    return "INFO label '6' == " + repr(lab["6"]) + "; all: " + ", ".join(f"{k}:{v}" for k, v in lab.items())


def t_kermany_untouched():
    # kermany entries must precede nih entries and keep their order; recheck count/layout
    kerm_idx = [i for i, e in enumerate(all_entries) if e["source"] == "kermany"]
    nih_idx = [i for i, e in enumerate(all_entries) if e["source"] == "nih"]
    assert max(kerm_idx) < min(nih_idx), "kermany entries are not all before nih entries"
    return "kermany block first, then nih block (verify_data.py covers Kermany content)"


def t_idempotent():
    h_img_before = adult_bytes_hash()
    r = subprocess.run([str(PY), str(ROOT / "scripts" / "prepare_adult.py")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"prepare_adult.py exit {r.returncode}\n{r.stderr[-800:]}"
    sha_after = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    h_img_after = adult_bytes_hash()
    assert sha_after == manifest_sha, f"manifest sha changed {manifest_sha} -> {sha_after}"
    assert h_img_after == h_img_before, "adult JPEG bytes changed after re-run"
    return f"manifest sha256 unchanged ({sha_after[:16]}...), adult tree hash unchanged ({h_img_after[:16]}...)"


for name, fn in [
    ("totals: 2240 = 1840 kermany + 400 nih, keys exact", t_totals),
    ("nih counts: train_adult/test_adult x normal/pneumonia = 100 each", t_counts),
    ("ids and paths unique across the whole manifest", t_ids),
    ("adult paths relative, forward-slash, exist, folder matches split+label, filename == id", t_paths),
    ("id scheme nih_<split>_<label>_0001..0100", t_gray_names),
    ("no stray files/dirs under adult/", t_strays),
    ("every adult JPEG is JPEG/RGB/224x224/q85/R==G==B", t_images),
    ("no pixel-identical images (train vs test adult, adult vs Kermany)", t_leakage),
    ("kermany entries precede nih entries", t_kermany_untouched),
    ("medmnist INFO: label 6 name is 'pneumonia'", t_label_name),
]:
    check(name, fn)

if not SKIP_SOURCE:
    check("(c) npz MD5 matches medmnist.INFO MD5_224", t_md5)
    print("-- streaming ChestMNIST npz (several minutes of decompression) ...", flush=True)
    try:
        source_scan()
        check("every adult image matches a unique source row in the CORRECT ChestMNIST split (test_adult->test, train_adult->train)", t_source_match)
        check("(a) labels: pneumonia => label[6]==1, normal => all 14 labels 0", t_source_labels)
        check("(b) matched rows reproduce an independent default_rng(42) sampling", t_resample)
    except Exception as e:  # noqa
        results.append(("source scan", False, repr(e)))
        print("FAIL  source scan:", repr(e))

if not SKIP_RERUN:
    check("idempotence: re-running prepare_adult.py leaves manifest + adult JPEG bytes unchanged", t_idempotent)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
