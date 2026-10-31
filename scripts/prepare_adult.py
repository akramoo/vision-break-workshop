#!/usr/bin/env python
"""D2: build the adult (NIH ChestX-ray14 via ChestMNIST 224) image sets for the workshop app.

Source: ChestMNIST 224 (MedMNIST v2, CC BY 4.0), derived from NIH ChestX-ray14 (CC0).
The .npz is expected at scripts/.cache/medmnist/chestmnist_224.npz (MD5 is checked against
medmnist.INFO); it is NOT downloaded here.

Outputs (under app/public/data/):
  adult/{train_adult,test_adult}/{normal,pneumonia}/<id>.jpg   224x224 RGB JPEG q85
  manifest.json   array of {id, path, label, split, source}; only entries with
                  source == 'nih' are replaced, all other sources (kermany) are kept as-is.

Labels: 'pneumonia' = pneumonia label == 1 (may co-occur with other findings);
        'normal'    = NO FINDING, i.e. all 14 ChestMNIST labels are 0.

Pools: 100 pneumonia + 100 no-finding from ChestMNIST `test` -> test_adult, and the same
from ChestMNIST `train` -> train_adult (used by the "mix in adult X-rays" fix).

Re-running is deterministic (seeded) and idempotent: the adult output dir is wiped and rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import medmnist
import numpy as np
from numpy.lib import format as npy_format
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "app" / "public" / "data"
OUT_DIR = DATA_DIR / "adult"
MANIFEST = DATA_DIR / "manifest.json"
NPZ_PATH = Path(__file__).resolve().parent / ".cache" / "medmnist" / "chestmnist_224.npz"

SIZE = 224
JPEG_QUALITY = 85
SOURCE = "nih"
PNEUMONIA_INDEX = 6  # the plan fixes this; asserted against the label names before sampling
SPLITS = ("train_adult", "test_adult")


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def check_labels() -> int:
    """Print the label table, look up 'pneumonia' by name and assert it is index 6."""
    names = {int(k): v for k, v in medmnist.INFO["chestmnist"]["label"].items()}
    print("ChestMNIST labels (index -> name):")
    for k in sorted(names):
        print(f"  {k:2d} -> {names[k]}")
    found = [k for k, v in names.items() if v.lower() == "pneumonia"]
    assert len(found) == 1, f"expected exactly one 'pneumonia' label, found {found}"
    assert found[0] == PNEUMONIA_INDEX, f"pneumonia is index {found[0]}, expected {PNEUMONIA_INDEX}"
    assert len(names) == 14
    print(f"Assertion OK: label {found[0]} == 'pneumonia'")
    return found[0]


def read_images(npz: zipfile.ZipFile, key: str, wanted: np.ndarray) -> dict[int, np.ndarray]:
    """Return {row_index: uint8[224,224]} for the wanted rows of `<key>.npy`.

    The npz is deflate-compressed (cannot be memory-mapped), so stream-decompress the
    member once, row by row, keeping only the wanted rows. Peak RAM stays a few MB.
    """
    wanted_set = {int(i) for i in wanted}
    out: dict[int, np.ndarray] = {}
    with npz.open(f"{key}.npy") as f:
        version = npy_format.read_magic(f)
        read_header = npy_format.read_array_header_1_0 if version == (1, 0) else npy_format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        assert dtype == np.uint8 and not fortran and shape[1:] == (SIZE, SIZE), (shape, dtype)
        row_bytes = SIZE * SIZE
        for i in range(shape[0]):
            buf = f.read(row_bytes)
            assert len(buf) == row_bytes, f"{key}: truncated at row {i}"
            if i in wanted_set:
                out[i] = np.frombuffer(buf, dtype=np.uint8).reshape(SIZE, SIZE)
                if len(out) == len(wanted_set):
                    break  # all needed rows read; skip the rest of the stream
    assert len(out) == len(wanted_set)
    return out


def draw(pool: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(pool, size=n, replace=False))  # sorted -> ids follow source index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="images per class per split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    t0 = time.time()

    pneu = check_labels()  # BEFORE any sampling

    expected_md5 = medmnist.INFO["chestmnist"]["MD5_224"]
    actual_md5 = md5_of(NPZ_PATH)
    print(f"\n{NPZ_PATH.name}: md5 {actual_md5} (expected {expected_md5})")
    assert actual_md5 == expected_md5, "MD5 mismatch -- re-download chestmnist_224.npz"

    rng = np.random.default_rng(args.seed)
    n = args.n
    with zipfile.ZipFile(NPZ_PATH) as npz:
        labels = {}
        for s in ("train", "val", "test"):
            with npz.open(f"{s}_labels.npy") as f:
                labels[s] = npy_format.read_array(f)
        for s, a in labels.items():
            assert a.shape[1] == 14, (s, a.shape)

        pools = {}
        print("\nCandidate pools (pneumonia / no-finding):")
        for s, a in labels.items():
            pools[s] = (np.flatnonzero(a[:, pneu] == 1), np.flatnonzero(a.sum(axis=1) == 0))
            print(f"  {s:5s} {len(pools[s][0]):6d} / {len(pools[s][1]):6d}  (of {len(a)})")

        # job lists: split -> [(label, source ChestMNIST split, row index)]
        chosen: dict[str, list[tuple[str, str, int]]] = {}
        for split, src in (("test_adult", "test"), ("train_adult", "train")):
            pos, neg = pools[src]
            if len(pos) >= n:
                pos_pick = [("pneumonia", src, int(i)) for i in draw(pos, n, rng)]
            else:
                # FALLBACK: too few pneumonia positives in this ChestMNIST split. Take them all
                # and draw the remainder from the val split, so the pool still has n positives.
                # (Not expected to trigger for test: ~280 positives exist.)
                short = n - len(pos)
                vpos = pools["val"][0]
                assert len(vpos) >= short, "not enough pneumonia positives even with val"
                print(f"\n!!! FALLBACK: {src} split has only {len(pos)} pneumonia positives (< {n}); "
                      f"drawing {short} from the val split for {split} !!!")
                pos_pick = [("pneumonia", src, int(i)) for i in pos]
                pos_pick += [("pneumonia", "val", int(i)) for i in draw(vpos, short, rng)]
            neg_pick = [("normal", src, int(i)) for i in draw(neg, n, rng)]
            chosen[split] = pos_pick + neg_pick

        # train and test come from disjoint ChestMNIST splits (NIH patient-level official split), so
        # overlap can only occur through the val fallback; check by (source split, row) anyway.
        keysets = [{(s, i) for _, s, i in jobs} for jobs in chosen.values()]
        assert not (keysets[0] & keysets[1]), "train_adult / test_adult overlap"

        shutil.rmtree(OUT_DIR, ignore_errors=True)  # clean so reruns leave no stale files
        entries: list[dict] = []
        for split in SPLITS:
            jobs = chosen[split]
            # read each source split's wanted rows with one streaming pass
            imgs: dict[tuple[str, int], np.ndarray] = {}
            for src in sorted({s for _, s, _ in jobs}):
                rows = np.array(sorted(i for _, s, i in jobs if s == src))
                print(f"\nReading {len(rows)} rows from {src}_images for {split} ...", flush=True)
                imgs.update({(src, i): a for i, a in read_images(npz, f"{src}_images", rows).items()})
            for label in ("pneumonia", "normal"):
                group = sorted((s, i) for lab, s, i in jobs if lab == label)  # stable id order
                for k, (src, idx) in enumerate(group, 1):
                    img_id = f"{SOURCE}_{split}_{label}_{k:04d}"
                    rel = f"adult/{split}/{label}/{img_id}.jpg"
                    dest = DATA_DIR / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    # Grayscale -> 3 channels (the model expects RGB); already 224x224.
                    Image.fromarray(imgs[(src, idx)], "L").convert("RGB").save(dest, "JPEG", quality=JPEG_QUALITY)
                    entries.append({"id": img_id, "path": rel, "label": label, "split": split, "source": SOURCE})
            print(f"  wrote {split}")

    verify(entries, n)

    # Merge-friendly manifest: replace only source == 'nih'; kermany (and others) stay in place first.
    others = []
    if MANIFEST.exists():
        others = [e for e in json.loads(MANIFEST.read_text("utf-8")) if e.get("source") != SOURCE]
    manifest = others + entries
    assert len({e["id"] for e in manifest}) == len(manifest), "id collision with other sources"
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWrote {MANIFEST} ({len(entries)} nih + {len(others)} other entries)")
    print(f"Total runtime: {time.time() - t0:.1f} s")


def verify(entries: list[dict], n: int) -> None:
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    counts: dict[tuple[str, str], int] = {}
    for e in entries:
        counts[(e["split"], e["label"])] = counts.get((e["split"], e["label"]), 0) + 1
    print("\nCounts (split / normal(=no finding) / pneumonia):")
    for split in SPLITS:
        print(f"  {split:12s} {counts.get((split, 'normal'), 0):4d} {counts.get((split, 'pneumonia'), 0):4d}")
    assert counts == {(s, lab): n for s in SPLITS for lab in ("normal", "pneumonia")}, counts

    total_bytes = 0
    for e in entries:
        p = DATA_DIR / e["path"]
        total_bytes += p.stat().st_size
        with Image.open(p) as im:
            assert im.format == "JPEG", f"{p}: format {im.format}"
            assert im.mode == "RGB", f"{p}: mode {im.mode}"
            assert im.size == (SIZE, SIZE), f"{p}: size {im.size}"
    on_disk = sum(1 for _ in OUT_DIR.rglob("*.jpg"))
    assert on_disk == len(entries), f"{on_disk} files on disk vs {len(entries)} manifest entries"
    print(f"\nAll {len(entries)} files are {SIZE}x{SIZE} RGB JPEG; ids unique; pools disjoint.")
    print(f"Total JPEG size: {total_bytes / 1e6:.2f} MB (mean {total_bytes / len(entries) / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
