"""Summarize T2I-CompBench vqa_result.json files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRIC_FILES = {
    "blip": "annotation_blip/vqa_result.json",
    "clip": "annotation_clip/vqa_result.json",
    "2d_spatial": "labels/annotation_obj_detection_2d/vqa_result.json",
    "3d_spatial": "labels/annotation_obj_detection_3d/vqa_result.json",
    "numeracy": "annotation_num/vqa_result.json",
    "3_in_1": "annotation_3_in_1/vqa_result.json",
}


def score_file(path: Path) -> tuple[int, float]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    values = [float(row["answer"]) for row in rows]
    return len(values), sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/t2i_compbench/eval_results"))
    parser.add_argument("--out", type=Path, default=Path("data/t2i_compbench/eval_results/summary.tsv"))
    args = parser.parse_args()

    rows = []
    for result_file in sorted(args.root.rglob("vqa_result.json")):
        rel = result_file.relative_to(args.root)
        if len(rel.parts) < 4:
            continue
        method, model, category = rel.parts[:3]
        suffix = "/".join(rel.parts[3:])
        metric = next((name for name, expected in METRIC_FILES.items() if suffix == expected), None)
        if metric is None:
            continue
        count, score = score_file(result_file)
        rows.append(
            {
                "rewrite_method": method,
                "model": model,
                "category": category,
                "metric": metric,
                "count": count,
                "score": f"{score:.6f}",
                "path": str(result_file),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rewrite_method", "model", "category", "metric", "count", "score", "path"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out} rows={len(rows)}")


if __name__ == "__main__":
    main()
