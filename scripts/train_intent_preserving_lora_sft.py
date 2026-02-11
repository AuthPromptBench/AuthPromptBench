from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


INTENT_PRESERVING_SYSTEM_PROMPT = (
    "You are an expert text-to-image prompt editor. "
    "Intent preservation is more important than brevity. "
    "Preserve all explicitly stated entities, quantities, actions, "
    "attributes, materials, relations, scenes, and styles. "
    "Do not add unsupported details or omit meaningful content. "
    "Return only the refined prompt."
)
ORIGINAL_REFINER_SYSTEM_PROMPT = (
    "You are an expert in text-to-image prompt refinement. "
    "Please keep your response concise and clear."
)
ORIGINAL_REFINER_INSTRUCTION = (
    "Convert to short description for text-to-image generation, "
    "Only include the description without any additional explanation."
)


@dataclass
class SFTExample:
    input: str
    output: str
    metadata: Dict[str, Any]


class PromptCaptionDataset:
    def __init__(self, examples: Sequence[SFTExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> SFTExample:
        return self.examples[index]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def iter_caption_outputs(
    item: Dict[str, Any],
    caption_models: Optional[set[str]],
    max_captions_per_model: Optional[int],
) -> Iterable[Tuple[str, str, int]]:
    captions_by_model = item.get("captions_by_model")
    if isinstance(captions_by_model, dict):
        for model, captions in captions_by_model.items():
            if caption_models and model not in caption_models:
                continue
            if not isinstance(captions, list):
                continue
            kept = 0
            for candidate_index, caption in enumerate(captions):
                output = clean_text(caption)
                if not output:
                    continue
                yield model, output, candidate_index
                kept += 1
                if max_captions_per_model is not None and kept >= max_captions_per_model:
                    break
        return

    if "enhanced_caption" in item:
        output = clean_text(item.get("enhanced_caption"))
        if output:
            yield "enhanced_caption", output, 0
        return

    captions = item.get("captions")
    if isinstance(captions, list):
        kept = 0
        for candidate_index, caption in enumerate(captions):
            output = clean_text(caption)
            if not output:
                continue
            yield "captions", output, candidate_index
            kept += 1
            if max_captions_per_model is not None and kept >= max_captions_per_model:
                break


def item_input_prompt(item: Dict[str, Any]) -> str:
    for key in ("prompt", "original_prompt", "input"):
        prompt = clean_text(item.get(key))
        if prompt:
            return prompt
    return ""


def load_jsonl_examples(
    path: Path,
    caption_models: Optional[set[str]],
    max_captions_per_model: Optional[int],
    caption_direction: str,
) -> List[SFTExample]:
    examples: List[SFTExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            input_prompt = item_input_prompt(item)
            if not input_prompt:
                continue
            for model, caption_text, candidate_index in iter_caption_outputs(
                item, caption_models, max_captions_per_model
            ):
                if caption_direction == "prompt_to_caption":
                    input_text = input_prompt
                    output_text = caption_text
                elif caption_direction == "caption_to_prompt":
                    input_text = caption_text
                    output_text = input_prompt
                else:
                    raise ValueError(f"Unsupported caption_direction: {caption_direction}")
                examples.append(
                    SFTExample(
                        input=input_text,
                        output=output_text,
                        metadata={
                            "path": str(path),
                            "line_no": line_no,
                            "id": item.get("id"),
                            "source": item.get("source"),
                            "caption_model": model,
                            "candidate_index": candidate_index,
                        },
                    )
                )
    return examples


def load_json_examples(
    path: Path,
    caption_models: Optional[set[str]],
    max_captions_per_model: Optional[int],
    caption_direction: str,
) -> List[SFTExample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Only list JSON caption files are supported: {path}")
    examples: List[SFTExample] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        input_prompt = item_input_prompt(item)
        if not input_prompt:
            continue
        for model, caption_text, candidate_index in iter_caption_outputs(
            item, caption_models, max_captions_per_model
        ):
            if caption_direction == "prompt_to_caption":
                input_text = input_prompt
                output_text = caption_text
            elif caption_direction == "caption_to_prompt":
                input_text = caption_text
                output_text = input_prompt
            else:
                raise ValueError(f"Unsupported caption_direction: {caption_direction}")
            examples.append(
                SFTExample(
                    input=input_text,
                    output=output_text,
                    metadata={
                        "path": str(path),
                        "row_index": index,
                        "id": item.get("id"),
                        "source": item.get("source"),
                        "caption_model": model,
                        "candidate_index": candidate_index,
                    },
                )
            )
    return examples


def load_examples(args: argparse.Namespace) -> List[SFTExample]:
    caption_models = set(args.caption_models) if args.caption_models else None
    examples: List[SFTExample] = []
    for raw_path in args.train_files:
        path = Path(raw_path)
        if path.suffix.lower() == ".jsonl":
            examples.extend(
                load_jsonl_examples(
                    path,
                    caption_models,
                    args.max_captions_per_model,
                    args.caption_direction,
                )
            )
        elif path.suffix.lower() == ".json":
            examples.extend(
                load_json_examples(
                    path,
                    caption_models,
                    args.max_captions_per_model,
                    args.caption_direction,
                )
            )
        else:
            raise ValueError(f"Unsupported training file extension: {path}")

    if args.dedupe:
        seen: set[Tuple[str, str]] = set()
        deduped: List[SFTExample] = []
        for example in examples:
            key = (example.input, example.output)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(example)
        examples = deduped

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("No SFT examples were loaded.")
    return examples


def apply_chat_template(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    def without_system_role() -> List[Dict[str, str]]:
        if not messages or messages[0].get("role") != "system" or len(messages) < 2:
            return messages
        system_content = messages[0].get("content", "")
        merged = [message.copy() for message in messages[1:]]
        if merged and merged[0].get("role") == "user":
            user_content = merged[0].get("content", "")
            merged[0]["content"] = f"{system_content}\n\n{user_content}".strip()
        return merged

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            return tokenizer.apply_chat_template(
                without_system_role(),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
    except Exception:
        return tokenizer.apply_chat_template(
            without_system_role(),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def is_mistral_common_backend(tokenizer) -> bool:
    return tokenizer.__class__.__name__ == "MistralCommonBackend"


def render_pair(tokenizer, input_text: str, output_text: str, prompt_style: str) -> Tuple[str, str]:
    if prompt_style == "intent":
        system_prompt = INTENT_PRESERVING_SYSTEM_PROMPT
        user_prompt = input_text
    elif prompt_style == "original":
        system_prompt = ORIGINAL_REFINER_SYSTEM_PROMPT
        user_prompt = (
            f"### Instruction:\n{ORIGINAL_REFINER_INSTRUCTION}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    else:
        raise ValueError(f"Unsupported prompt_style: {prompt_style}")

    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": output_text},
    ]
    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    if is_mistral_common_backend(tokenizer):
        full_text = prompt_text + output_text + (tokenizer.eos_token or "")
    else:
        full_text = apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)
        if not full_text.startswith(prompt_text):
            full_text = prompt_text + output_text + (tokenizer.eos_token or "")
    return prompt_text, full_text


def encode_example(
    tokenizer,
    example: SFTExample,
    max_length: int,
    prompt_style: str,
) -> Dict[str, torch.Tensor]:
    import torch

    prompt_text, full_text = render_pair(tokenizer, example.input, example.output, prompt_style)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    label_start = min(len(prompt_ids), len(full_ids))
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        full_ids = full_ids[overflow:]
        label_start = max(0, label_start - overflow)

    input_ids = torch.tensor(full_ids, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:label_start] = -100
    if (labels != -100).sum().item() == 0:
        labels[-1] = input_ids[-1]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class SFTCollator:
    def __init__(self, tokenizer, max_length: int, prompt_style: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_style = prompt_style

    def __call__(self, examples: List[SFTExample]) -> Dict[str, torch.Tensor]:
        import torch

        encoded = [
            encode_example(self.tokenizer, example, self.max_length, self.prompt_style)
            for example in examples
        ]
        max_len = max(item["input_ids"].numel() for item in encoded)
        pad_id = self.tokenizer.pad_token_id
        batch: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in encoded:
            pad_len = max_len - item["input_ids"].numel()
            batch["input_ids"].append(
                torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_id)
            )
            batch["attention_mask"].append(
                torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
            )
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
        return {key: torch.stack(value, dim=0) for key, value in batch.items()}


def lora_state_dict(module) -> Dict[str, "torch.Tensor"]:
    import torch

    if hasattr(module, "model"):
        module = module.model
    if hasattr(module, "module"):
        module = module.module

    state: Dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if "lora_" in name or "modules_to_save" in name or "prompt_embeddings" in name:
            state[name.replace(".module.", ".")] = value.detach().cpu()
    return state


def save_checkpoint(policy, output_path: str, step: int, args: argparse.Namespace) -> None:
    import torch

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    state = lora_state_dict(policy)
    if not state:
        raise ValueError("No LoRA weights found to save.")
    torch.save(
        {
            "state_dict": state,
            "global_step": step,
            "hyper_parameters": {
                "model_name": args.model_name,
                "lr": args.lr,
                "max_new_tokens": args.max_new_tokens,
                "use_lora": True,
                "prompt_style": args.prompt_style,
                "system_prompt": (
                    INTENT_PRESERVING_SYSTEM_PROMPT
                    if args.prompt_style == "intent"
                    else ORIGINAL_REFINER_SYSTEM_PROMPT
                ),
                "instruction": ORIGINAL_REFINER_INSTRUCTION if args.prompt_style == "original" else None,
                "train_files": args.train_files,
                "caption_models": args.caption_models,
                "max_captions_per_model": args.max_captions_per_model,
                "caption_direction": args.caption_direction,
            },
        },
        output_path,
    )
    print(f"saved_checkpoint={output_path} state_dict_keys={len(state)}", flush=True)

    if args.adapter_output_dir:
        os.makedirs(args.adapter_output_dir, exist_ok=True)
        model_to_save = policy.model.module if hasattr(policy.model, "module") else policy.model
        model_to_save.save_pretrained(args.adapter_output_dir)
        policy.tokenizer.save_pretrained(args.adapter_output_dir)
        print(f"saved_adapter_dir={args.adapter_output_dir}", flush=True)


def distributed_context(args: argparse.Namespace):
    import torch
    import torch.distributed as dist

    env_world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    env_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    env_local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", str(args.device))))
    distributed = env_world_size > 1
    visible_devices = torch.cuda.device_count()
    device_index = 0 if visible_devices == 1 else env_local_rank
    torch.cuda.set_device(device_index)
    if distributed and not dist.is_initialized():
        try:
            dist.init_process_group(
                backend=args.dist_backend,
                init_method="env://",
                device_id=torch.device(f"cuda:{device_index}"),
            )
        except TypeError:
            dist.init_process_group(backend=args.dist_backend, init_method="env://")
    return distributed, env_rank, env_world_size, env_local_rank, device_index


def rank0_print(rank: int, text: str) -> None:
    if rank == 0:
        print(text, flush=True)


class Mistral3SFTPolicy:
    @staticmethod
    def resolve_model_path(model_name: str) -> str:
        if Path(model_name).exists():
            return model_name

        from huggingface_hub import snapshot_download

        allow_patterns = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "model-*.safetensors",
            "params.json",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "tekken.json",
            "chat_template.jinja",
        ]
        try:
            return snapshot_download(
                repo_id=model_name,
                allow_patterns=allow_patterns,
                local_files_only=True,
            )
        except Exception:
            return snapshot_download(
                repo_id=model_name,
                allow_patterns=allow_patterns,
                resume_download=True,
            )

    def __init__(
        self,
        model_name: str,
        lr: float,
        max_new_tokens: int,
        use_lora: bool,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration, MistralCommonBackend

        self.model_name = model_name
        self.lr = lr
        self.max_new_tokens = max_new_tokens
        model_path = self.resolve_model_path(model_name)
        if os.environ.get("RANK", "0") == "0":
            print(f"resolved_mistral3_model_path={model_path}", flush=True)
        self.tokenizer = MistralCommonBackend.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = Mistral3ForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            quantization_config=FineGrainedFP8Config(dequantize=True),
        )
        if use_lora:
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],
            )
            self.model = get_peft_model(self.model, peft_config)
        if device and device != "auto":
            self.model = self.model.to(device)
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def train(self) -> None:
        self.model.train()

    def trainable_parameters(self):
        return (param for param in self.model.parameters() if param.requires_grad)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        if not state:
            raise ValueError(f"Checkpoint {checkpoint_path} has an empty state_dict.")
        incompatible = self.model.load_state_dict(state, strict=False)
        print(
            f"Loaded refiner checkpoint from {checkpoint_path}. "
            f"missing_keys={len(incompatible.missing_keys)} unexpected_keys={len(incompatible.unexpected_keys)}",
            flush=True,
        )


class GenericSFTPolicy:
    """Small PEFT wrapper shared by Llama, Qwen, and Mistral-style causal LMs."""

    def __init__(self, model_name: str, lr: float, max_new_tokens: int, checkpoint_path: str | None, device: str):
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.lr = lr
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],
            ),
        ).to(device)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)

    def train(self) -> None:
        self.model.train()

    def trainable_parameters(self):
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)


def build_sft_policy(args: argparse.Namespace, device: str):
    if args.model_name.startswith("mistralai/Ministral-3"):
        return Mistral3SFTPolicy(
            model_name=args.model_name,
            lr=args.lr,
            max_new_tokens=args.max_new_tokens,
            use_lora=True,
            checkpoint_path=args.resume_checkpoint,
            device=device,
        )

    return GenericSFTPolicy(
        model_name=args.model_name,
        lr=args.lr,
        max_new_tokens=args.max_new_tokens,
        checkpoint_path=args.resume_checkpoint,
        device=device,
    )


def train(args: argparse.Namespace) -> None:
    examples = load_examples(args)
    if args.dry_run:
        print(f"loaded_examples={len(examples)}", flush=True)
        target_steps = args.max_steps if args.max_steps is not None else len(examples) * args.num_epochs
        print(f"target_steps={target_steps}", flush=True)
        for example in examples[: args.preview_count]:
            print(
                json.dumps(
                    {
                        "input": example.input,
                        "output": example.output,
                        "prompt_style": args.prompt_style,
                        "metadata": example.metadata,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    distributed, rank, world_size, local_rank, device_index = distributed_context(args)
    device = f"cuda:{device_index}"
    rank0_print(rank, f"loaded_examples={len(examples)}")
    rank0_print(
        rank,
        f"distributed={distributed} rank={rank} world_size={world_size} "
        f"local_rank={local_rank} device={device}",
    )
    policy = build_sft_policy(args, device)
    policy.tokenizer.padding_side = "right"
    if distributed:
        ddp_find_unused_parameters = args.model_name.startswith("mistralai/Ministral-3")
        rank0_print(rank, f"ddp_find_unused_parameters={ddp_find_unused_parameters}")
        policy.model = DistributedDataParallel(
            policy.model,
            device_ids=[device_index],
            output_device=device_index,
            find_unused_parameters=ddp_find_unused_parameters,
        )
    policy.train()

    dataset = PromptCaptionDataset(examples)
    collator = SFTCollator(policy.tokenizer, args.max_length, args.prompt_style)
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
        drop_last=False,
    )
    target_steps = args.max_steps if args.max_steps is not None else len(loader) * args.num_epochs
    rank0_print(rank, f"local_steps_per_epoch={len(loader)} target_steps={target_steps}")
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    while step < target_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = policy.model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            raw_loss = float((loss.detach() * args.gradient_accumulation_steps).cpu())
            step += 1
            if rank == 0 and (step % args.log_every == 0 or step == 1):
                print(f"step={step} loss={raw_loss:.6f}", flush=True)
            if rank == 0 and args.save_every and step % args.save_every == 0:
                save_checkpoint(policy, args.output_path, step, args)
            if step >= target_steps:
                break
        epoch += 1

    if step % args.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(list(policy.trainable_parameters()), args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        save_checkpoint(policy, args.output_path, step, args)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT for IntentPreservingAutoModelLLMRefiner using prompt-caption pairs."
    )
    parser.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--train_files", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--adapter_output_dir", default=None)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--prompt_style", choices=["intent"], default="intent")
    parser.add_argument("--caption_models", nargs="*", default=None)
    parser.add_argument("--max_captions_per_model", type=int, default=None)
    parser.add_argument(
        "--caption_direction",
        choices=["prompt_to_caption", "caption_to_prompt"],
        default="prompt_to_caption",
        help="Use original prompt as input and caption as output, or reverse the pair.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dedupe", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dist_backend", default="nccl")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--preview_count", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
