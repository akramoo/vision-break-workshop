"""Independent verification of D1 (app/public/data/manifest.json + kermany JPEGs).
Run: .venv/Scripts/python scripts/tests/verify_data.py
Exit code is non-zero if any check fails."""
import hashlib
import io
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "app" / "public" / "data"
MANIFEST = DATA / "manifest.json"

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((name, True, detail or ""))
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    except AssertionError as e:
        results.append((name, False, str(e)))
        print(f"FAIL  {name}: {e}")
    except Exception as e:  # noqa
        results.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"FAIL  {name}: {type(e).__name__}: {e}")


raw = MANIFEST.read_bytes()
print("manifest.json sha256:", hashlib.sha256(raw).hexdigest(), "bytes:", len(raw))
all_entries = json.loads(raw)
assert isinstance(all_entries, list), "manifest must be a list"
# D2 adds source=nih entries; these checks cover the Kermany entries only.
manifest = [e for e in all_entries if e.get("source") == "kermany"]

EXPECTED = {
    ("train", "normal"): 255, ("train", "pneumonia"): 765,
    ("val", "normal"): 45, ("val", "pneumonia"): 135,
    ("val_official", "normal"): 8, ("val_official", "pneumonia"): 8,
    ("test", "normal"): 234, ("test", "pneumonia"): 390,
}


def t_keys():
    bad = [e for e in manifest if set(e.keys()) != {"id", "path", "label", "split", "source"}]
    assert not bad, f"{len(bad)} entries with wrong keys, first: {bad[0]}"
    assert all(isinstance(v, str) for e in manifest for v in e.values()), "non-string value"
    return f"{len(manifest)} entries"


def t_labels():
    bad = [e for e in manifest if e["label"] not in ("normal", "pneumonia")]
    assert not bad, f"bad labels: {bad[:2]}"
    badsp = [e for e in manifest if e["split"] not in ("train", "val", "val_official", "test")]
    assert not badsp, f"bad splits: {badsp[:2]}"
    bads = [e for e in manifest if e["source"] != "kermany"]
    assert not bads, f"non-kermany source: {bads[:2]}"


def t_counts():
    c = Counter((e["split"], e["label"]) for e in manifest)
    assert dict(c) == EXPECTED, f"got {dict(c)}"
    return str(dict(sorted(c.items())))


def t_ids_unique():
    ids = [e["id"] for e in manifest]
    assert len(ids) == len(set(ids)), f"{len(ids) - len(set(ids))} duplicate ids"
    train_ids = {e["id"] for e in manifest if e["split"] == "train"}
    val_ids = {e["id"] for e in manifest if e["split"] == "val"}
    assert not (train_ids & val_ids), "train/val id overlap"


def t_paths():
    paths = [e["path"] for e in manifest]
    assert len(paths) == len(set(paths)), "duplicate paths"
    for e in manifest:
        p = e["path"]
        assert "\\" not in p, f"backslash in {p}"
        assert not p.startswith("/") and ":" not in p and ".." not in p.split("/"), f"not relative: {p}"
        parts = p.split("/")
        assert len(parts) == 4 and parts[0] == "kermany", f"unexpected layout: {p}"
        assert parts[1] == e["split"], f"split folder mismatch: {e}"
        assert parts[2] == e["label"], f"label folder mismatch: {e}"
        assert parts[3] == e["id"] + ".jpg", f"filename != id: {e}"
        assert (DATA / p).is_file(), f"missing file {p}"


def t_strays():
    on_disk = {p.relative_to(DATA).as_posix() for p in (DATA / "kermany").rglob("*") if p.is_file()}
    listed = {e["path"] for e in manifest}
    assert on_disk == listed, f"stray={sorted(on_disk - listed)[:3]} missing={sorted(listed - on_disk)[:3]}"
    jpgs = [p for p in on_disk if p.endswith(".jpg")]
    assert len(jpgs) == len(manifest), f"{len(jpgs)} jpg vs {len(manifest)} entries"
    return f"{len(jpgs)} jpg files == {len(manifest)} entries, no other files"


def expected_q85_tables():
    # Re-encode a test image with PIL at q85 and read back the quantization tables.
    buf = io.BytesIO()
    arr = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(buf, "JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).quantization


Q85 = expected_q85_tables()
pixel_hash = {}  # id -> sha256 of decoded pixels


def t_images():
    total = 0
    bad_q = []
    for e in manifest:
        p = DATA / e["path"]
        total += p.stat().st_size
        with Image.open(p) as im:
            assert im.format == "JPEG", f"{e['path']}: format {im.format}"
            assert im.mode == "RGB", f"{e['path']}: mode {im.mode}"
            assert im.size == (224, 224), f"{e['path']}: size {im.size}"
            q = im.quantization
            if q[0] != Q85[0] or q.get(1) != Q85.get(1):
                bad_q.append((e["path"], q[0][:3]))
            im.load()
            pixel_hash[e["id"]] = hashlib.sha256(np.asarray(im).tobytes()).hexdigest()
    assert not bad_q, f"{len(bad_q)} files with non-q85 quant tables, e.g. {bad_q[:2]}"
    return f"all RGB 224x224 JPEG; luma q-table[0:3]={Q85[0][:3]} matches PIL q85; total {total/1e6:.2f} MB"


def t_size():
    total = sum((DATA / e["path"]).stat().st_size for e in manifest)
    assert total < 60 * 1024 * 1024, f"total {total} bytes >= 60MB"
    return f"{total/1e6:.2f} MB"


def t_ratio():
    def n(s, l):
        return sum(1 for e in manifest if e["split"] == s and e["label"] == l)
    tn, tp, vn, vp = n("train", "normal"), n("train", "pneumonia"), n("val", "normal"), n("val", "pneumonia")
    assert abs(tp / tn - 3.0) < 0.1, f"train ratio {tp/tn}"
    assert abs(vp / vn - 3.0) < 0.1, f"val ratio {vp/vn}"
    assert abs(vn / (tn + vn) - 0.15) < 0.01 and abs(vp / (tp + vp) - 0.15) < 0.01, "val is not ~15% stratified"
    return f"train 1:{tp/tn:.2f}, val 1:{vp/vn:.2f}, val frac normal {vn/(tn+vn):.3f} pneu {vp/(tp+vp):.3f}"


def t_leakage():
    assert pixel_hash, "images check must run first"
    by = {s: {} for s in ("train", "val", "val_official", "test")}
    for e in manifest:
        by[e["split"]].setdefault(pixel_hash[e["id"]], []).append(e["id"])
    tv = set(by["train"]) | set(by["val"])
    leak = tv & set(by["test"])
    assert not leak, f"{len(leak)} identical-pixel images in train/val AND test, e.g. {[by['test'][h] for h in list(leak)[:2]]}"
    leak2 = set(by["val_official"]) & set(by["test"])
    assert not leak2, f"{len(leak2)} val_official images identical to test"
    tvl = set(by["train"]) & set(by["val"])
    assert not tvl, f"{len(tvl)} identical-pixel images in both train and val"
    msg = []
    lo = set(by["val_official"]) & tv
    if lo:
        msg.append(f"NOTE {len(lo)} val_official dup train/val")
    for s, d in by.items():
        dups = sum(len(v) - 1 for v in d.values())
        if dups:
            msg.append(f"NOTE {dups} intra-{s} pixel duplicates")
    return "train/val vs test, train vs val, val_official vs test disjoint; " + ("; ".join(msg) if msg else "no intra-split dups")


def t_gray():
    rng = random.Random(1234)
    for e in rng.sample(manifest, 50):
        with Image.open(DATA / e["path"]) as im:
            a = np.asarray(im.convert("RGB")).astype(np.int16)
        d = max(np.abs(a[..., 0] - a[..., 1]).max(), np.abs(a[..., 1] - a[..., 2]).max())
        assert d == 0, f"{e['path']}: R/G/B differ by up to {d}"
    return "50/50 sampled images have R==G==B for every pixel"


for name, fn in [
    ("manifest entries have exactly keys id,path,label,split,source", t_keys),
    ("label/split/source values valid", t_labels),
    ("split counts match D1", t_counts),
    ("ids unique, train/val id-disjoint", t_ids_unique),
    ("paths unique, relative, forward-slash, exist, folder matches split/label", t_paths),
    ("no stray files on disk", t_strays),
    ("every JPEG RGB 224x224 and quant tables == PIL q85", t_images),
    ("total size < 60MB", t_size),
    ("class ratio ~1:3 and val ~15% stratified", t_ratio),
    ("no pixel-content leakage", t_leakage),
    ("grayscale-as-RGB (R==G==B)", t_gray),
]:
    check(name, fn)

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
