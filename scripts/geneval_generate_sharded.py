"""Generate images for GenEval metadata using this project's T2I loaders.

The output layout matches the official GenEval evaluator:

<outdir>/
    00000/
        metadata.jsonl
        samples/
            00000.png
            00001.png
            ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generation_backend import generate_image, get_device, load_model


def load_metadata(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_indices(indices: str | None, total: int) -> List[int] | None:
    if not indices:
        return None
    parsed: List[int] = []
    for part in indices.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            parsed.extend(range(int(start_text), int(end_text) + 1))
        else:
            parsed.append(int(part))
    unique = sorted(set(parsed))
    for index in unique:
        if index < 0 or index >= total:
            raise ValueError(f"Prompt index {index} outside [0, {total})")
    return unique


def shard_indices(total: int, rank: int, world_size: int, indices: str | None) -> List[int]:
    selected = parse_indices(indices, total)
    if selected is None:
        selected = list(range(total))
    return selected[rank::world_size]


def resolve_rank(args: argparse.Namespace) -> tuple[int, int, int, int]:
    local_proc_rank = int(os.environ.get("SLURM_PROCID", args.rank))
    rank_offset = int(os.environ.get("AP_RANK_OFFSET", "0"))
    rank = local_proc_rank + rank_offset
    default_world_size = int(os.environ.get("SLURM_NTASKS", args.world_size))
    world_size = int(os.environ.get("AP_WORLD_SIZE", default_world_size))
    return rank, world_size, local_proc_rank, rank_offset


def sample_path(outdir: Path, prompt_index: int, sample_index: int) -> Path:
    return outdir / f"{prompt_index:05d}" / "samples" / f"{sample_index:05d}.png"


def write_metadata(outdir: Path, prompt_index: int, metadata: Dict[str, Any]) -> None:
    prompt_dir = outdir / f"{prompt_index:05d}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    with open(prompt_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/geneval/prompts/evaluation_metadata.jsonl"),
        help="GenEval evaluation_metadata.jsonl.",
    )
    parser.add_argument(
        "--model",
        choices=["flux1", "sd21", "sd3", "comat21", "pixart-sigma", "infi", "showo2"],
        required=True,
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma/semicolon separated indices or ranges.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    metadata_rows = load_metadata(args.metadata)
    rank, world_size, local_proc_rank, rank_offset = resolve_rank(args)
    indices = shard_indices(len(metadata_rows), rank, world_size, args.indices)
    device = get_device()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[geneval worker {rank}/{world_size} local_proc_rank={local_proc_rank} "
        f"rank_offset={rank_offset}] model={args.model} device={device} prompts={len(indices)}",
        flush=True,
    )

    if not indices:
        timing_dir = args.outdir / "_timings"
        timing_dir.mkdir(parents=True, exist_ok=True)
        timing_path = timing_dir / f"worker_{rank:05d}.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "rank": rank,
                    "world_size": world_size,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "model": args.model,
                    "metadata": str(args.metadata),
                    "outdir": str(args.outdir),
                    "samples_per_prompt": args.samples_per_prompt,
                    "generated_images": 0,
                    "skipped_existing_images": 0,
                    "timing_seconds": {"model_load": 0.0, "worker_total": 0.0},
                    "item_timings": [],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[geneval worker {rank}] empty shard; skipped model load", flush=True)
        return

    load_start = time.perf_counter()
    pipe = load_model(args.model, device)
    load_seconds = time.perf_counter() - load_start
    print(f"[geneval worker {rank}] model_load_seconds={load_seconds:.2f}", flush=True)

    generated = 0
    skipped = 0
    timings = []
    worker_start = time.perf_counter()
    for offset, prompt_index in enumerate(indices, start=1):
        metadata = metadata_rows[prompt_index]
        prompt = metadata["prompt"]
        write_metadata(args.outdir, prompt_index, metadata)
        (args.outdir / f"{prompt_index:05d}" / "samples").mkdir(parents=True, exist_ok=True)

        prompt_start = time.perf_counter()
        for sample_index in range(args.samples_per_prompt):
            out_path = sample_path(args.outdir, prompt_index, sample_index)
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            seed = args.seed_offset + sample_index
            print(
                f"[geneval worker {rank}] generating prompt_index={prompt_index} "
                f"sample_index={sample_index} seed={seed}",
                flush=True,
            )
            image = generate_image(pipe, args.model, prompt, device, seed)
            image.save(out_path)
            generated += 1
            print(f"[geneval worker {rank}] saved {out_path}", flush=True)

        prompt_seconds = time.perf_counter() - prompt_start
        timings.append({"prompt_index": prompt_index, "seconds": round(prompt_seconds, 4)})
        print(
            f"[geneval worker {rank}] {offset}/{len(indices)} prompt_index={prompt_index} "
            f"seconds={prompt_seconds:.2f}",
            flush=True,
        )

    timing_dir = args.outdir / "_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    timing_path = timing_dir / f"worker_{rank:05d}.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rank": rank,
                "world_size": world_size,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "metadata": str(args.metadata),
                "outdir": str(args.outdir),
                "samples_per_prompt": args.samples_per_prompt,
                "generated_images": generated,
                "skipped_existing_images": skipped,
                "timing_seconds": {
                    "model_load": round(load_seconds, 4),
                    "worker_total": round(time.perf_counter() - worker_start + load_seconds, 4),
                },
                "item_timings": timings,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[geneval worker {rank}] done generated={generated} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
