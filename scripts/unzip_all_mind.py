#!/usr/bin/env python3
from pathlib import Path
from zipfile import ZipFile

'''
Unzip all MIND dataset files.

Usage
-----
$ python scripts/unzip_all_mind.py
'''

mind_dir = Path("data/mind")

for zip_path in mind_dir.glob("*.zip"):
    out_dir = mind_dir / zip_path.stem
    out_dir.mkdir(exist_ok=True)
    print(f"Extracting {zip_path.name} → {out_dir}/")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)