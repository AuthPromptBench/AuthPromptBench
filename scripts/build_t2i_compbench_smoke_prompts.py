"""Build one-prompt T2I-CompBench smoke prompt files."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SMOKE_CATEGORIES = [
    "color_val",
    "non_spatial_val",
    "spatial_val",
    "3d_spatial_val",
    "numeracy_val",
    "complex_val",
]


def first_line(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No non-empty lines in {path}")


def write_one(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(first_line(src) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-dir",
        type=Path,
        default=Path("data/t2i_compbench/prompts/official_dataset"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("data/t2i_compbench/prompts/smoke_dataset"),
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name in SMOKE_CATEGORIES:
        write_one(args.official_dir / f"{name}.txt", args.outdir / f"{name}_smoke.txt")

    # 3-in-1 reads these exact official names from data_path.
    complex_prompt = first_line(args.official_dir / "complex_val.txt")
    for name in ["complex_val.txt", "complex_val_action.txt", "complex_val_spatial.txt"]:
        (args.outdir / name).write_text(complex_prompt + "\n", encoding="utf-8")
    (args.outdir / "complex_val_spatialaction.txt").write_text("", encoding="utf-8")

    new_objects = args.official_dir / "new_objects.txt"
    if new_objects.exists():
        shutil.copy2(new_objects, args.outdir / "new_objects.txt")

    print(f"wrote smoke prompts to {args.outdir}")


if __name__ == "__main__":
    main()
