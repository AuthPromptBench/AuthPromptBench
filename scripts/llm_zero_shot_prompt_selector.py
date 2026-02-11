#!/usr/bin/env python3
"""Zero-shot LLM selector for routing between original and rewritten prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_PATH = ROOT / "data/benchmark/auth_prompt_V1_0_0.json"
DEFAULT_REWRITTEN_PATH = ROOT / "data/benchmark/refined_prompts/qwen3.json"
DEFAULT_OUTPUT_JSONL = ROOT / "data/benchmark/analysis/zero_shot_selector_choices.jsonl"
DEFAULT_ROUTED_PATH = ROOT / "data/benchmark/refined_prompts/zero_shot_selector_routed.json"

OPENROUTER_ENV = "OPENROUTER_API_KEY"
DEEPSEEK_ENV = "DEEPSEEK_API_KEY"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"

SYSTEM_PROMPT = "You are a careful judge of text-to-image prompt quality. Return valid JSON only."

SELECTOR_PROMPT_TEMPLATE = """You are selecting the best prompt for image generation.

Choose between:
1. the original user prompt;
2. the rewritten prompt.

Evaluate:
- preservation of explicitly stated entities, attributes, actions and relations;
- omission of important concepts;
- unsupported semantic changes;
- resolution of ambiguity when needed;
- executability by a text-to-image model.

Do not always prefer longer or more fluent prompts.
Do not penalize the original prompt merely because it is informal.
When candidates are nearly equivalent, prefer the original prompt.

Original user prompt:
{original_prompt}

Rewritten prompt:
{rewritten_prompt}

Return JSON only:
{{
  "choice": "original | rewritten",
  "confidence": 0.0
}}"""

CHOICES = {"original", "rewritten"}


@dataclass
class SelectorResult:
    choice: str
    confidence: float
    raw_response: str
    parse_error: str | None = None


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    return data


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def load_existing_choices(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("choice") not in CHOICES or "rewritten_prompt" not in row:
                raise ValueError(
                    f"{path} contains choices from an incompatible selector schema; "
                    "rerun with --overwrite."
                )
            out[row["id"]] = row
    return out


def validate_position_alignment(name: str, reference: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    if len(reference) != len(rows):
        raise ValueError(f"{name} length mismatch: {len(rows)} vs {len(reference)}")
    mismatches = [
        (idx, reference[idx].get("id"), rows[idx].get("id"))
        for idx in range(len(reference))
        if reference[idx].get("id") != rows[idx].get("id")
    ]
    if mismatches:
        raise ValueError(f"{name} id mismatch at benchmark position: {mismatches[:5]}")


def selector_prompt(original_prompt: str, rewritten_prompt: str) -> str:
    return SELECTOR_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        rewritten_prompt=rewritten_prompt,
    )


def parse_selector_response(text: str) -> SelectorResult:
    raw = text.strip()
    candidates = [raw]
    if raw.startswith("```"):
        stripped = raw.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        candidates.append(stripped)

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    last_error = "no JSON object found"
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue
        if not isinstance(parsed, dict):
            last_error = "JSON response is not an object"
            continue
        choice = str(parsed.get("choice", "")).strip().lower()
        confidence = parsed.get("confidence", 0.0)
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0
        confidence_value = min(1.0, max(0.0, confidence_value))
        if choice not in CHOICES:
            return SelectorResult(
                choice="original",
                confidence=0.0,
                raw_response=raw,
                parse_error=f"invalid choice: {choice!r}",
            )
        return SelectorResult(choice=choice, confidence=confidence_value, raw_response=raw)

    return SelectorResult(choice="original", confidence=0.0, raw_response=raw, parse_error=last_error)


def call_openrouter(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(base_url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(retry_sleep)
    raise RuntimeError(f"selector request failed: {last_error}")


def call_deepseek(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    reasoning_effort: str | None,
    thinking: str,
) -> str:
    try:
        from openai import APIConnectionError, APIError, APITimeoutError, OpenAI
    except ImportError as exc:
        return call_deepseek_with_requests(
            prompt=prompt,
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_sleep=retry_sleep,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
        )

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    request: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort and reasoning_effort.lower() not in {"omit", "none", "disabled"}:
        request["reasoning_effort"] = reasoning_effort
    if thinking != "omit":
        request["extra_body"] = {"thinking": {"type": thinking}}

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(**request)
            return response.choices[0].message.content or ""
        except (APIConnectionError, APITimeoutError, APIError, KeyError, IndexError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(retry_sleep)
    raise RuntimeError(f"DeepSeek selector request failed: {last_error}")


def call_deepseek_with_requests(
    prompt: str,
    api_key: str,
    model: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    retries: int,
    retry_sleep: float,
    reasoning_effort: str | None,
    thinking: str,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning_effort and reasoning_effort.lower() not in {"omit", "none", "disabled"}:
        payload["reasoning_effort"] = reasoning_effort
    if thinking != "omit":
        payload["extra_body"] = {"thinking": {"type": thinking}}

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
        except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(retry_sleep)
    raise RuntimeError(f"DeepSeek requests selector failed: {last_error}")


class HFSelectorClient:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        device_map: str,
        torch_dtype: str,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = None
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32

        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": True,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            tokenized = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,
            )
        tokenized = {
            key: value.to(self.model.device) if isinstance(value, torch.Tensor) else value
            for key, value in tokenized.items()
        }
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": self.max_tokens,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
            output_ids = self.model.generate(**tokenized, **generation_kwargs)
        generated_ids = output_ids[:, tokenized["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def selected_prompt(choice: str, original_prompt: str, rewritten_prompt: str) -> str:
    return rewritten_prompt if choice == "rewritten" else original_prompt


def make_choice_row(
    index: int,
    base: dict[str, Any],
    rewritten_item: dict[str, Any],
    result: SelectorResult,
    model: str,
) -> dict[str, Any]:
    original_prompt = base.get("prompt") or ""
    rewritten_prompt = rewritten_item.get("refined_prompt") or rewritten_item.get("prompt") or ""
    return {
        "sample_id": index + 1,
        "id": base.get("id"),
        "benchmark_index": base.get("index"),
        "user_type": base.get("user_type"),
        "apbench_class": base.get("class"),
        "intent": base.get("intent"),
        "sentence_intent": base.get("sentence_intent"),
        "choice": result.choice,
        "confidence": result.confidence,
        "selected_prompt": selected_prompt(result.choice, original_prompt, rewritten_prompt),
        "original_prompt": original_prompt,
        "rewritten_prompt": rewritten_prompt,
        "selector_model": model,
        "raw_response": result.raw_response,
        "parse_error": result.parse_error,
    }


def run_selector_for_index(
    index: int,
    benchmark: list[dict[str, Any]],
    rewritten_items: list[dict[str, Any]],
    args: argparse.Namespace,
    api_key: str | None,
    deepseek_api_key: str | None,
    hf_client: "HFSelectorClient | None",
) -> dict[str, Any]:
    base = benchmark[index]
    original_prompt = base.get("prompt") or ""
    rewritten_prompt = rewritten_items[index].get("refined_prompt") or rewritten_items[index].get("prompt") or ""
    prompt = selector_prompt(original_prompt, rewritten_prompt)

    try:
        if args.backend == "hf":
            if hf_client is None:
                raise RuntimeError("HF selector client was not initialized.")
            raw_response = hf_client.generate(prompt)
        elif args.backend == "deepseek":
            raw_response = call_deepseek(
                prompt=prompt,
                api_key=deepseek_api_key or "",
                model=args.model,
                base_url=args.deepseek_base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
                reasoning_effort=args.deepseek_reasoning_effort,
                thinking=args.deepseek_thinking,
            )
        else:
            raw_response = call_openrouter(
                prompt=prompt,
                api_key=api_key or "",
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retries=args.retries,
                retry_sleep=args.retry_sleep,
            )
        result = parse_selector_response(raw_response)
    except Exception as exc:
        result = SelectorResult(choice="original", confidence=0.0, raw_response="", parse_error=str(exc))

    return make_choice_row(index, base, rewritten_items[index], result, args.model)


def write_routed_prompts(
    benchmark: list[dict[str, Any]],
    choices_by_id: dict[str, dict[str, Any]],
    output_path: Path,
    selector_model: str,
) -> None:
    routed = []
    for item in benchmark:
        row = choices_by_id.get(item["id"])
        if row is None:
            raise ValueError(f"missing selector choice for id={item['id']}")
        new_item = item.copy()
        new_item["refined_prompt"] = row["selected_prompt"]
        new_item["refiner_type"] = "zero_shot_selector_routed"
        new_item["selector_choice"] = row["choice"]
        new_item["selector_confidence"] = row["confidence"]
        new_item["selector_model"] = selector_model
        routed.append(new_item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(routed, f, ensure_ascii=False, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a zero-shot LLM selector over original and rewritten prompts.")
    parser.add_argument("--benchmark_path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--rewritten_path", type=Path, default=DEFAULT_REWRITTEN_PATH)
    parser.add_argument("--output_jsonl", type=Path, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--routed_output_path", type=Path, default=DEFAULT_ROUTED_PATH)
    parser.add_argument("--backend", choices=["openrouter", "deepseek", "hf"], default="openrouter")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api_key_env", default=OPENROUTER_ENV)
    parser.add_argument("--deepseek_base_url", default=DEFAULT_DEEPSEEK_BASE_URL)
    parser.add_argument("--deepseek_api_key_env", default=DEEPSEEK_ENV)
    parser.add_argument("--deepseek_reasoning_effort", default="high")
    parser.add_argument("--deepseek_thinking", choices=["enabled", "disabled", "omit"], default="enabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=80)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent API requests for openrouter/deepseek backends.")
    parser.add_argument("--device_map", default="auto", help="HF backend device_map.")
    parser.add_argument("--torch_dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of samples to process.")
    parser.add_argument("--start", type=int, default=0, help="0-based start position.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing selector JSONL.")
    parser.add_argument("--dry_run", action="store_true", help="Print the first selector prompt and exit without API calls.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.backend == "deepseek" and args.model == DEFAULT_MODEL:
        args.model = DEFAULT_DEEPSEEK_MODEL
    benchmark = load_json(args.benchmark_path)
    rewritten_items = load_json(args.rewritten_path)
    validate_position_alignment("rewritten prompts", benchmark, rewritten_items)

    if args.dry_run:
        idx = args.start
        print(selector_prompt(
            benchmark[idx].get("prompt") or "",
            rewritten_items[idx].get("refined_prompt") or rewritten_items[idx].get("prompt") or "",
        ))
        return

    api_key = os.environ.get(args.api_key_env)
    if args.backend == "openrouter" and not api_key:
        raise ValueError(f"{args.api_key_env} is required.")
    deepseek_api_key = os.environ.get(args.deepseek_api_key_env)
    if args.backend == "deepseek" and not deepseek_api_key:
        raise ValueError(f"{args.deepseek_api_key_env} is required.")
    hf_client = None
    if args.backend == "hf":
        hf_client = HFSelectorClient(
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
        )

    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    existing = load_existing_choices(args.output_jsonl)
    end = len(benchmark) if args.limit is None else min(len(benchmark), args.start + args.limit)
    pending_indices = [index for index in range(args.start, end) if benchmark[index]["id"] not in existing]
    if args.backend == "hf" or args.concurrency <= 1:
        for index in pending_indices:
            row = run_selector_for_index(
                index=index,
                benchmark=benchmark,
                rewritten_items=rewritten_items,
                args=args,
                api_key=api_key,
                deepseek_api_key=deepseek_api_key,
                hf_client=hf_client,
            )
            write_jsonl_row(args.output_jsonl, row)
            existing[row["id"]] = row
            print(
                f"[{index + 1}/{len(benchmark)}] id={row['id']} choice={row['choice']} "
                f"confidence={row['confidence']:.3f}",
                flush=True,
            )
    else:
        workers = max(1, args.concurrency)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_selector_for_index,
                    index,
                    benchmark,
                    rewritten_items,
                    args,
                    api_key,
                    deepseek_api_key,
                    None,
                ): index
                for index in pending_indices
            }
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                index = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    base = benchmark[index]
                    result = SelectorResult(choice="original", confidence=0.0, raw_response="", parse_error=str(exc))
                    row = make_choice_row(index, base, rewritten_items[index], result, args.model)
                write_jsonl_row(args.output_jsonl, row)
                existing[row["id"]] = row
                completed += 1
                print(
                    f"[{completed}/{total} completed; sample {index + 1}/{len(benchmark)}] "
                    f"id={row['id']} choice={row['choice']} confidence={row['confidence']:.3f}",
                    flush=True,
                )

    if len(existing) == len(benchmark):
        write_routed_prompts(benchmark, existing, args.routed_output_path, args.model)
        print(f"Wrote routed prompts to {args.routed_output_path}", flush=True)
    else:
        print(
            f"Selector choices incomplete: {len(existing)}/{len(benchmark)}. "
            "Routed prompt file will be written after all samples are selected.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
