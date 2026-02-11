"""Create category-specific T2I-CompBench sample views from combined samples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.t2i_compbench_generate_sharded import filename_prompt


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_manifest(path: Path) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_prompts(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


def link_or_check(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return
        raise FileExistsError(f"{dst} already exists and does not point to {src}")
    dst.symlink_to(src)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--combined-image-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    parser.add_argument(
        "--view-prompt-source",
        choices=("source", "manifest"),
        default="source",
        help=(
            "Use source prompts for category-view filenames by default. "
            "Use manifest to expose official prompt filenames while linking "
            "to images generated from --prompts."
        ),
    )
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    prompts = load_prompts(args.prompts)
    if prompts is not None and len(prompts) != len(rows):
        raise ValueError(f"{args.prompts} has {len(prompts)} prompts, expected {len(rows)}")
    combined_samples = args.combined_image_dir / "samples"
    if not combined_samples.is_dir():
        raise NotADirectoryError(combined_samples)

    created = 0
    for row in rows:
        category = str(row["category"])
        global_index = int(row["global_index"])
        category_index = int(row["category_index"])
        source_prompt = prompts[global_index] if prompts is not None else str(row["prompt"])
        view_prompt = str(row["prompt"]) if args.view_prompt_source == "manifest" else source_prompt
        category_samples = args.out_root / category / "samples"
        category_samples.mkdir(parents=True, exist_ok=True)

        for sample_index in range(args.samples_per_prompt):
            global_question_id = global_index * args.samples_per_prompt + sample_index
            category_question_id = category_index * args.samples_per_prompt + sample_index
            src = combined_samples / f"{filename_prompt(source_prompt)}_{global_question_id:06d}.png"
            dst = category_samples / f"{filename_prompt(view_prompt)}_{category_question_id:06d}.png"
            link_or_check(src, dst)
            created += 1

    print(f"materialized {created} category sample links under {args.out_root}")


if __name__ == "__main__":
    main()
