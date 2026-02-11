"""Fetch official T2I-CompBench prompt files into this repo.

The official evaluator expects the dataset files from
Karine-Huang/T2I-CompBench/examples/dataset. This script keeps a local copy
under data/t2i_compbench/prompts/official_dataset so the CompBench workflow is
isolated from GenEval.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


DATASET_FILES = [
    "3d_spatial.txt",
    "3d_spatial_train.txt",
    "3d_spatial_val.txt",
    "color.txt",
    "color_train.txt",
    "color_val.txt",
    "color_val_seen.txt",
    "color_val_unseen.txt",
    "complex.txt",
    "complex_train.txt",
    "complex_train_action.txt",
    "complex_train_spatial.txt",
    "complex_train_spatialaction.txt",
    "complex_val.txt",
    "complex_val_action.txt",
    "complex_val_spatial.txt",
    "complex_val_spatialaction.txt",
    "new_objects.txt",
    "non_spatial.txt",
    "non_spatial_train.txt",
    "non_spatial_val.txt",
    "numeracy.txt",
    "numeracy_train.txt",
    "numeracy_val.txt",
    "shape.txt",
    "shape_train.txt",
    "shape_val.txt",
    "shape_val_seen.txt",
    "shape_val_unseen.txt",
    "spatial.txt",
    "spatial_train.txt",
    "spatial_val.txt",
    "texture.txt",
    "texture_train.txt",
    "texture_val.txt",
    "texture_val_seen.txt",
    "texture_val_unseen.txt",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/t2i_compbench/prompts/official_dataset"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/Karine-Huang/T2I-CompBench/main/examples/dataset"
    for name in DATASET_FILES:
        out_path = args.outdir / name
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue
        url = f"{base}/{name}"
        print(f"[fetch] {url}")
        with urllib.request.urlopen(url) as response:
            data = response.read()
        out_path.write_bytes(data)
        print(f"[write] {out_path} {len(data)} bytes")


if __name__ == "__main__":
    main()
