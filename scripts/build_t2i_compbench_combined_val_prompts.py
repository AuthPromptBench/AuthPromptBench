"""Build a combined T2I-CompBench val prompt file and category manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VAL_CATEGORIES = [
    "color_val",
    "shape_val",
    "texture_val",
    "non_spatial_val",
    "spatial_val",
    "3d_spatial_val",
    "numeracy_val",
    "complex_val",
]


def load_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


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
        default=Path("data/t2i_compbench/prompts/combined_dataset"),
    )
    parser.add_argument("--name", default="all_val")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.outdir / f"{args.name}.txt"
    manifest_path = args.outdir / f"{args.name}_manifest.jsonl"

    global_index = 0
    rows: list[dict[str, int | str]] = []
    with prompt_path.open("w", encoding="utf-8") as prompt_file:
        for category in VAL_CATEGORIES:
            prompts = load_prompts(args.official_dir / f"{category}.txt")
            for category_index, prompt in enumerate(prompts):
                prompt_file.write(prompt + "\n")
                rows.append(
                    {
                        "global_index": global_index,
                        "category": category,
                        "category_index": category_index,
                        "prompt": prompt,
                    }
                )
                global_index += 1

    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        for row in rows:
            manifest_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {global_index} prompts to {prompt_path}")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
