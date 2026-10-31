"""Helper for verify_embeddings.mjs: decode JPEGs with PIL and dump raw RGB bytes.

usage: python -I pil_dump_rgb.py files.json out.u8
files.json is a JSON list of image paths; out.u8 gets n * 224*224*3 uint8 bytes (RGB, row-major).
"""
import json
import sys

from PIL import Image

files = json.load(open(sys.argv[1], encoding="utf-8"))
with open(sys.argv[2], "wb") as out:
    for f in files:
        im = Image.open(f).convert("RGB")
        assert im.size == (224, 224), (f, im.size)
        out.write(im.tobytes())
print(f"dumped {len(files)} images")
