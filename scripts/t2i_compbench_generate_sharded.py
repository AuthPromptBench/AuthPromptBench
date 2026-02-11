"""Generate flat T2I-CompBench evaluation images.

T2I-CompBench evaluators expect one category at a time under:

<outdir>/
    samples/
        <prompt>_000000.png
        <prompt>_000001.png
        ...

The last six digits are the question_id used by the official scripts. This
layout is intentionally separate from GenEval's nested prompt directories.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.generation_backend import generate_image, get_device, load_model


def load_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)
    return prompts


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


def filename_prompt(prompt: str) -> str:
    safe = prompt.strip().replace("_", " ")
    safe = safe.replace("/", " or ")
    safe = re.sub(r"\s+", " ", safe)
    return safe[:180]


def image_path(outdir: Path, prompt: str, question_id: int) -> Path:
    return outdir / "samples" / f"{filename_prompt(prompt)}_{question_id:06d}.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=["flux1", "sd21", "sd3", "comat21", "pixart-sigma", "infi", "showo2"],
        required=True,
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--indices", default=None, help="Comma/semicolon separated indices or ranges.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    rank, world_size, local_proc_rank, rank_offset = resolve_rank(args)
    indices = shard_indices(len(prompts), rank, world_size, args.indices)
    device = get_device()
    samples_dir = args.outdir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[t2i-compbench worker {rank}/{world_size} local_proc_rank={local_proc_rank} "
        f"rank_offset={rank_offset}] model={args.model} device={device} prompts={len(indices)}",
        flush=True,
    )

    if not indices:
        return

    load_start = time.perf_counter()
    pipe = load_model(args.model, device)
    load_seconds = time.perf_counter() - load_start
    print(f"[t2i-compbench worker {rank}] model_load_seconds={load_seconds:.2f}", flush=True)

    generated = 0
    skipped = 0
    manifest_rows = []
    worker_start = time.perf_counter()
    for offset, prompt_index in enumerate(indices, start=1):
        prompt = prompts[prompt_index]
        prompt_start = time.perf_counter()
        for sample_index in range(args.samples_per_prompt):
            question_id = prompt_index * args.samples_per_prompt + sample_index
            out_path = image_path(args.outdir, prompt, question_id)
            manifest_rows.append(
                {
                    "prompt_index": prompt_index,
                    "sample_index": sample_index,
                    "question_id": question_id,
                    "prompt": prompt,
                    "file_name": out_path.name,
                }
            )
            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue
            seed = args.seed_offset + sample_index
            print(
                f"[t2i-compbench worker {rank}] generating prompt_index={prompt_index} "
                f"sample_index={sample_index} question_id={question_id} seed={seed}",
                flush=True,
            )
            image = generate_image(pipe, args.model, prompt, device, seed)
            image.save(out_path)
            generated += 1

        prompt_seconds = time.perf_counter() - prompt_start
        print(
            f"[t2i-compbench worker {rank}] {offset}/{len(indices)} prompt_index={prompt_index} "
            f"seconds={prompt_seconds:.2f}",
            flush=True,
        )

    timing_dir = args.outdir / "_timings"
    timing_dir.mkdir(parents=True, exist_ok=True)
    with (timing_dir / f"worker_{rank:05d}.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "rank": rank,
                "world_size": world_size,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "prompts": str(args.prompts),
                "outdir": str(args.outdir),
                "samples_per_prompt": args.samples_per_prompt,
                "generated_images": generated,
                "skipped_existing_images": skipped,
                "timing_seconds": {
                    "model_load": round(load_seconds, 4),
                    "worker_total": round(time.perf_counter() - worker_start + load_seconds, 4),
                },
                "manifest": manifest_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[t2i-compbench worker {rank}] done generated={generated} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
