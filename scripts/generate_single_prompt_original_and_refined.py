#!/usr/bin/env python3
"""Refine one prompt and generate images for original/refined variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics.refine import REFINERS
from scripts.generation_backend import generate_image, get_device, load_model


DEFAULT_PROMPT = (
    "a blue-eyed black siamese cat wearing a bowtie, by makoto shinkai, "
    "greg rutkowski, artstation, high detailed, cgsociety"
)


def maybe_clear_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def save_image(image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def refine_prompt(prompt: str, refiner_type: str, temperature: float, checkpoint: str | None) -> tuple[str, dict]:
    if refiner_type not in REFINERS:
        raise ValueError(f"Unknown refiner type: {refiner_type}")
    spec = REFINERS[refiner_type]
    if spec.needs_checkpoint and not checkpoint:
        raise ValueError("A checkpoint is required for LoRA refiners")

    started = time.perf_counter()
    builder_args = argparse.Namespace(model_name=None, checkpoint=checkpoint)
    refiner = spec.builder(builder_args)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    if hasattr(refiner, "refine_prompt"):
        raw = refiner.refine_prompt(prompt)
    else:
        generated = refiner.generate([prompt], temperature=temperature)
        raw = generated[0] if isinstance(generated, list) else generated
    refine_seconds = time.perf_counter() - started
    refined = (raw or "").strip()
    if refined.lower().startswith(("i cannot", "i can't", "i can’t")):
        refined = prompt

    del refiner
    maybe_clear_cuda()
    return refined, {
        "refiner_type": refiner_type,
        "checkpoint_path": checkpoint,
        "raw_refined_prompt": raw,
        "load_seconds": round(load_seconds, 4),
        "refine_seconds": round(refine_seconds, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate original and refined images for one prompt.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--refiner_type", default="llama3_lora")
    parser.add_argument("--models", nargs="+", default=["flux1", "pixart-sigma"])
    parser.add_argument("--images_per_variant", type=int, default=3)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--output_root",
        default="outputs/single_prompt",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = get_device()

    refined_prompt, refine_meta = refine_prompt(args.prompt, args.refiner_type, args.temperature, args.checkpoint)
    variants = {
        "original": args.prompt,
        args.refiner_type: refined_prompt,
    }

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "prompt": args.prompt,
        "refined_prompt": refined_prompt,
        "refine": refine_meta,
        "models": args.models,
        "images_per_variant": args.images_per_variant,
        "seed_start": args.seed_start,
        "outputs": [],
    }
    (output_root / "prompts.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Original prompt: {args.prompt}", flush=True)
    print(f"Refined prompt: {refined_prompt}", flush=True)
    print(f"Output root: {output_root}", flush=True)

    for model in args.models:
        load_start = time.perf_counter()
        pipe = load_model(model, device)
        load_seconds = time.perf_counter() - load_start
        print(f"[{model}] model_load_seconds={load_seconds:.2f}", flush=True)

        for variant, prompt in variants.items():
            for seed in range(args.seed_start, args.seed_start + args.images_per_variant):
                out_path = output_root / variant / model / f"seed_{seed:05d}.png"
                if out_path.exists() and not args.overwrite:
                    print(f"[{model}/{variant}] skip existing {out_path}", flush=True)
                    continue
                started = time.perf_counter()
                image = generate_image(pipe, model, prompt, device, seed)
                save_image(image, out_path)
                seconds = time.perf_counter() - started
                metadata["outputs"].append(
                    {
                        "variant": variant,
                        "model": model,
                        "seed": seed,
                        "path": str(out_path),
                        "seconds": round(seconds, 4),
                    }
                )
                print(f"[{model}/{variant}] seed={seed} seconds={seconds:.2f} path={out_path}", flush=True)

        del pipe
        maybe_clear_cuda()
        (output_root / "prompts.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote metadata to {output_root / 'prompts.json'}", flush=True)


if __name__ == "__main__":
    main()
