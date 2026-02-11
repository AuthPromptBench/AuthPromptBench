# Prompt routing

## Zero-shot routing

`scripts/llm_zero_shot_prompt_selector.py` asks an API or local LLM to choose strictly between the original prompt and one rewritten prompt. It is resumable and defaults to the original prompt when a response cannot be parsed.

Create the rewritten prompt file first, for example with `python metrics/refine.py --refiner_type qwen3`. The command below then compares that file with the original benchmark prompts:

```bash
python scripts/llm_zero_shot_prompt_selector.py \
  --rewritten_path data/benchmark/refined_prompts/qwen3.json \
  --dry_run
```

Remove `--dry_run` to run the configured backend. Existing output from an older or incompatible selector schema is rejected; pass `--overwrite` to start a clean result file.

## Learned per-generator routers

`scripts/train_selector.py` trains an independent gain regressor for every requested image generator. It uses exactly 80% of the training set for fitting and 20% for model/threshold selection. The test set is evaluated only after validation selection.

```bash
python scripts/train_selector.py \
  --train_benchmark data/train/router_train.json \
  --test_benchmark data/benchmark/auth_prompt_V1_0_0.json \
  --train_default train_original \
  --train_rewrite train_intent_preserving_lora \
  --test_default original \
  --test_rewrite llama3_lora
```

For each generator, the command exports `checkpoints/<generator>/selector.joblib`. Each checkpoint contains the fitted regressor, selected feature set, validation-selected threshold, split IDs, and test predictions/decisions. Feature preprocessing and cached embeddings are written beside the reports so a run can be audited.
