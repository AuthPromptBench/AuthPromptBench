"""Refine APBench-compatible prompts with traditional, local, LoRA, or API methods."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/benchmark/auth_prompt_V1_0_0.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/benchmark/refined_prompts"


@dataclass(frozen=True)
class RefinerSpec:
    builder: Callable[[argparse.Namespace], object]
    needs_checkpoint: bool = False


def _traditional(name: str):
    def build(_: argparse.Namespace):
        if name == "pae":
            from models.components.refiner.pae_refiner import PAE_refiner

            checkpoint = ROOT / "checkpoints/PAE/sd21/actor_step3000.pt"
            return PAE_refiner(ckpt_path=str(checkpoint))
        from models.components.refiner.refiner import RefinerModel

        model = "microsoft/Promptist" if name == "promptist" else "pag"
        return RefinerModel(pretrained_model_name=model)

    return build


def _local(family: str, use_lora: bool):
    def build(args: argparse.Namespace):
        from models.components.refiner.llmrefiner import (
            IntentPreservingAutoModelLLMRefiner,
            IntentPreservingMistral3Refiner,
            IntentPreservingQwen3LLMRefiner,
        )

        defaults = {
            "llama3": "meta-llama/Llama-3.1-8B-Instruct",
            "mistral3": "mistralai/Ministral-3-8B-Instruct-2512",
            "mistral7b": "mistralai/Mistral-7B-Instruct-v0.2",
            "qwen3": "Qwen/Qwen3-8B",
        }
        model_name = args.model_name or defaults[family]
        if family == "mistral3":
            refiner = IntentPreservingMistral3Refiner(model_name=model_name, use_fp8=False, use_lora=use_lora)
        elif family == "qwen3":
            refiner = IntentPreservingQwen3LLMRefiner(model_name=model_name, use_lora=use_lora)
        else:
            refiner = IntentPreservingAutoModelLLMRefiner(model_name=model_name, use_lora=use_lora)
        if use_lora:
            refiner.load_checkpoint(args.checkpoint)
        return refiner

    return build


def _openrouter(model: str):
    def build(_: argparse.Namespace):
        from models.components.refiner.closed_llm_refiner import OpenRouterIntentPreservingRefiner

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for API refiners")
        return OpenRouterIntentPreservingRefiner(api_key=api_key, model=model)

    return build


REFINERS = {
    "promptist": RefinerSpec(_traditional("promptist")),
    "pag": RefinerSpec(_traditional("pag")),
    "pae_sd21": RefinerSpec(_traditional("pae")),
    "llama3": RefinerSpec(_local("llama3", False)),
    "mistral3": RefinerSpec(_local("mistral3", False)),
    "mistral7b": RefinerSpec(_local("mistral7b", False)),
    "qwen3": RefinerSpec(_local("qwen3", False)),
    "llama3_lora": RefinerSpec(_local("llama3", True), True),
    "mistral3_lora": RefinerSpec(_local("mistral3", True), True),
    "mistral7b_lora": RefinerSpec(_local("mistral7b", True), True),
    "qwen3_lora": RefinerSpec(_local("qwen3", True), True),
    "gpt-4o-mini": RefinerSpec(_openrouter("openai/gpt-4o-mini")),
    "gemini-2.5-flash-lite": RefinerSpec(_openrouter("google/gemini-2.5-flash-lite")),
    "grok-3-mini": RefinerSpec(_openrouter("x-ai/grok-3-mini")),
    "deepseek-chat": RefinerSpec(_openrouter("deepseek/deepseek-chat-v3.1")),
}


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")
    return data


def save_rows(path: Path, rows: list[dict], refined: list[str], method: str, checkpoint: str | None) -> None:
    if len(rows) != len(refined):
        raise ValueError(f"Expected {len(rows)} outputs, received {len(refined)}")
    output = []
    for row, text in zip(rows, refined):
        item = dict(row)
        value = (text or "").strip()
        if value.lower().startswith(("i cannot", "i can't", "i can’t")):
            value = item.get("prompt", "")
        item["refined_prompt"] = value
        item["refiner_type"] = method
        if checkpoint:
            item["refiner_checkpoint"] = checkpoint
        output.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refiner_type", required=True, choices=sorted(REFINERS))
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model_name", help="Optional Hugging Face model override")
    parser.add_argument("--checkpoint", help="LoRA checkpoint for *_lora methods")
    args = parser.parse_args()

    spec = REFINERS[args.refiner_type]
    if spec.needs_checkpoint and not args.checkpoint:
        parser.error("--checkpoint is required for *_lora methods")
    rows = load_rows(args.input)
    prompts = [str(row.get("prompt") or "") for row in rows]
    refiner = spec.builder(args)
    output = args.output or DEFAULT_OUTPUT_DIR / f"{args.refiner_type}.json"

    def save(refined: list[str]) -> None:
        save_rows(output, rows, refined, args.refiner_type, args.checkpoint)

    refiner.generate_and_save(prompts, save)
    print(f"Saved {len(rows)} refined prompts to {output}")


if __name__ == "__main__":
    main()
