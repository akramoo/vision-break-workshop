#!/usr/bin/env python
"""D1: build the Kermany chest X-ray image sets for the workshop app.

Source: HF dataset `hf-vision/chest-xray-pneumonia` (Kermany et al. 2018, CC BY 4.0).

Outputs (under app/public/data/):
  kermany/{train,val,val_official,test}/{normal,pneumonia}/<id>.jpg   224x224 RGB JPEG q85
  manifest.json   array of {id, path, label, split, source}; only entries with
                  source == 'kermany' are replaced, other sources are preserved.

Re-running is deterministic: the kermany output dir is wiped and rebuilt, and the
manifest is identical for the same seed / arguments.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "app" / "public" / "data"
OUT_DIR = DATA_DIR / "kermany"
MANIFEST = DATA_DIR / "manifest.json"
CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "hf"

DATASET = "hf-vision/chest-xray-pneumonia"
LABEL_NAMES = ["NORMAL", "PNEUMONIA"]
LABELS = ["normal", "pneumonia"]  # index == dataset label id
SIZE = 224
JPEG_QUALITY = 85
SOURCE = "kermany"

EXPECTED_OFFICIAL = {"val_official": (8, 8), "test": (234, 390)}


def convert(img: Image.Image) -> Image.Image:
    # Grayscale -> 3 channels (the model expects RGB), then a plain 224x224 resize.
    # Deliberately NOT aspect-preserving: no center crop or padding, so the whole
    # radiograph stays visible (the originals vary in aspect ratio and a crop would cut
    # off lung edges). The stretch is mild and identical for every image, and the same
    # transform is applied to train and test, so the classifier sees a consistent view.
    return img.convert("L").convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)


def pick_split(dsd, *names: str):
    for n in names:
        if n in dsd:
            return dsd[n]
    raise KeyError(f"none of {names} in dataset splits {list(dsd.keys())}")


def sample_indices(labels: np.ndarray, label: int, n: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.flatnonzero(labels == label)
    if n > len(pool):
        raise ValueError(f"requested {n} of label {label} but only {len(pool)} available")
    return np.sort(rng.choice(pool, size=n, replace=False))  # sorted -> stable id order


def val_split(idx: np.ndarray, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx), both sorted; val gets round(frac * n)."""
    n_val = round(len(idx) * frac)
    chosen = rng.choice(len(idx), size=n_val, replace=False)
    mask = np.zeros(len(idx), dtype=bool)
    mask[chosen] = True
    return idx[~mask], idx[mask]


def write_images(ds, jobs: list[tuple[str, str, int]], tag: str) -> list[dict]:
    """jobs: (split, label_name, source_index) in output order -> manifest entries."""
    entries = []
    counters: dict[tuple[str, str], int] = {}
    for k, (split, label, src_idx) in enumerate(jobs, 1):
        n = counters[(split, label)] = counters.get((split, label), 0) + 1
        img_id = f"{SOURCE}_{split}_{label}_{n:04d}"
        rel = f"{SOURCE}/{split}/{label}/{img_id}.jpg"
        dest = DATA_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        convert(ds[int(src_idx)]["image"]).save(dest, "JPEG", quality=JPEG_QUALITY)
        entries.append({"id": img_id, "path": rel, "label": label, "split": split, "source": SOURCE})
        if k % 200 == 0 or k == len(jobs):
            print(f"  [{tag}] {k}/{len(jobs)} images written", flush=True)
    return entries


def verify(entries: list[dict], expected: dict[tuple[str, str], int]) -> None:
    ids = [e["id"] for e in entries]
    assert len(ids) == len(set(ids)), "duplicate ids"
    paths = [e["path"] for e in entries]
    assert len(paths) == len(set(paths)), "duplicate paths"

    counts: dict[tuple[str, str], int] = {}
    for e in entries:
        key = (e["split"], e["label"])
        counts[key] = counts.get(key, 0) + 1
    print("\nCounts (split / normal / pneumonia):")
    for split in ("train", "val", "val_official", "test"):
        print(f"  {split:13s} {counts.get((split, 'normal'), 0):5d} {counts.get((split, 'pneumonia'), 0):5d}")
    assert counts == expected, f"counts {counts} != expected {expected}"

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
    print(f"\nAll {len(entries)} files are {SIZE}x{SIZE} RGB JPEG; ids unique.")
    print(f"Total JPEG size: {total_bytes / 1e6:.1f} MB (mean {total_bytes / len(entries) / 1e3:.1f} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-normal", type=int, default=300)
    ap.add_argument("--n-pneumonia", type=int, default=900)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Loading {DATASET} (cache: {CACHE_DIR}) ...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dsd = load_dataset(DATASET, cache_dir=str(CACHE_DIR))
    print(f"Splits: { {k: len(v) for k, v in dsd.items()} }")

    train_ds = pick_split(dsd, "train")
    val_ds = pick_split(dsd, "validation", "val")
    test_ds = pick_split(dsd, "test")
    for name, d in (("train", train_ds), ("validation", val_ds), ("test", test_ds)):
        names = d.features["label"].names
        assert names == LABEL_NAMES, f"{name}: label names {names} != {LABEL_NAMES}"
    print(f"Label names OK: {LABEL_NAMES}")

    rng = np.random.default_rng(args.seed)
    train_labels = np.asarray(train_ds["label"])
    wanted = {0: args.n_normal, 1: args.n_pneumonia}

    jobs: list[tuple[str, str, int]] = []
    expected: dict[tuple[str, str], int] = {}
    per_label_train: dict[int, np.ndarray] = {}
    per_label_val: dict[int, np.ndarray] = {}
    for lab in (0, 1):  # fixed order -> deterministic RNG consumption
        sub = sample_indices(train_labels, lab, wanted[lab], rng)
        per_label_train[lab], per_label_val[lab] = val_split(sub, args.val_frac, rng)

    # Job order: train, val (normal then pneumonia), val_official, test. Within each
    # (split, label) the ids follow ascending source index, so they are stable.
    for split, groups in (("train", per_label_train), ("val", per_label_val)):
        for lab in (0, 1):
            jobs += [(split, LABELS[lab], i) for i in groups[lab]]
            expected[(split, LABELS[lab])] = len(groups[lab])
    entries = []
    print("\nWriting train/val ...", flush=True)
    shutil.rmtree(OUT_DIR, ignore_errors=True)  # clean so reruns leave no stale files
    entries += write_images(train_ds, jobs, "train+val")

    for split, d in (("val_official", val_ds), ("test", test_ds)):
        labels = np.asarray(d["label"])
        sub_jobs = [(split, LABELS[lab], i) for lab in (0, 1) for i in np.flatnonzero(labels == lab)]
        print(f"Writing {split} ...", flush=True)
        entries += write_images(d, sub_jobs, split)
        for lab in (0, 1):
            expected[(split, LABELS[lab])] = int((labels == lab).sum())

    # The plan fixes these counts for the default arguments.
    if (args.n_normal, args.n_pneumonia, args.val_frac) == (300, 900, 0.15):
        want = {("train", "normal"): 255, ("train", "pneumonia"): 765,
                ("val", "normal"): 45, ("val", "pneumonia"): 135,
                ("val_official", "normal"): 8, ("val_official", "pneumonia"): 8,
                ("test", "normal"): 234, ("test", "pneumonia"): 390}
        assert expected == want, f"split sizes {expected} != plan {want}"

    verify(entries, expected)

    # Merge-friendly manifest: replace only source == 'kermany'; keep everything else.
    others = []
    if MANIFEST.exists():
        others = [e for e in json.loads(MANIFEST.read_text("utf-8")) if e.get("source") != SOURCE]
    manifest = entries + others
    assert len({e["id"] for e in manifest}) == len(manifest), "id collision with other sources"
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWrote {MANIFEST} ({len(entries)} kermany + {len(others)} other entries)")


if __name__ == "__main__":
    main()
