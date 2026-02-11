# Prompt editing and training

## Editing methods

`metrics/refine.py` accepts APBench-compatible JSON arrays containing a `prompt` field.

Traditional methods:

```bash
python metrics/refine.py --refiner_type promptist
python metrics/refine.py --refiner_type pag
python metrics/refine.py --refiner_type pae_sd21
```

Local models:

```bash
python metrics/refine.py --refiner_type llama3
python metrics/refine.py --refiner_type mistral3
python metrics/refine.py --refiner_type mistral7b
python metrics/refine.py --refiner_type qwen3
```

LoRA variants use the same editing instruction and require an explicit checkpoint:

```bash
python metrics/refine.py --refiner_type qwen3_lora --checkpoint checkpoints/qwen3/editor.pt
```

API methods require `OPENROUTER_API_KEY` and include `gpt-4o-mini`, `gemini-2.5-flash-lite`, `grok-3-mini`, and `deepseek-chat`.

Use `--input` and `--output` for custom APBench-compatible files.

## LoRA SFT

Training data can be JSON or JSONL prompt-caption pairs. The trainer supports Llama, Qwen, Mistral-style causal LMs, and a dedicated Ministral loader.

```bash
python scripts/train_intent_preserving_lora_sft.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --train_files data/train/pairs.jsonl \
  --output_path checkpoints/llama3/editor.pt \
  --dedupe --num_epochs 1
```

Use `--dry_run` to validate data and rendered examples without loading a model.

## CoMat

CoMat training consumes JSONL rows with a `text` or `prompt` field:

```bash
python scripts/train_comat.py \
  --prompts data/train/prompts.jsonl \
  --model_config configs/comat/model/sd21.yaml \
  --output_dir checkpoints/comat
```

