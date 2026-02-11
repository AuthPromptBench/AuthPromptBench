# Reproducibility guide

This supplement contains the benchmark metadata, implementation, configurations, and generic experiment entry points. Large model weights, generated images, API responses, private credentials, and third-party evaluator repositories are intentionally excluded.

## 1. Environment

Use Python 3.10 or newer. Install a CUDA- or ROCm-compatible PyTorch and torchvision build first, then install the pinned direct dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url YOUR_PLATFORM_SPECIFIC_INDEX
pip install -r requirements.txt
```

Copy `.env.example` to a file outside the submitted archive or export credentials from the shell. API credentials are needed only for API-backed refiners/selectors.

## 2. Validate the supplement

```bash
python -m json.tool data/benchmark/auth_prompt_V1_0_0.json >/dev/null
python metrics/refine.py --help
python scripts/train_intent_preserving_lora_sft.py --help
python scripts/llm_zero_shot_prompt_selector.py --help
```

## 3. Prompt editing

```bash
python metrics/refine.py --refiner_type qwen3
```

Expected output: `data/benchmark/refined_prompts/qwen3.json`, containing 2,048 records aligned by `id` with the original benchmark.

## 4. Image generation

```bash
python metrics/generate.py --model sd21 --refiner_type qwen3 --images_per_prompt 8
```

Expected output: eight numbered images under each `data/benchmark/generated_images/qwen3/sd21/prompt_<index>/` directory. Generation is resumable because existing seed files are skipped.

## 5. APBench evaluation

```bash
python -m metrics.eval \
  --images_dir data/benchmark/generated_images/qwen3/sd21 \
  --result_dir data/benchmark/results/qwen3/sd21
python metrics/print_res.py data/benchmark/results/qwen3/sd21
```

The evaluator writes `clip.jsonl`, `qwen.jsonl`, and `merged_result.jsonl`. The result summarizer reports aggregate and APBench subgroup metrics.

## 6. Prompt routing

Zero-shot routing compares exactly two candidates: original and rewritten.

```bash
python scripts/llm_zero_shot_prompt_selector.py \
  --rewritten_path data/benchmark/refined_prompts/qwen3.json \
  --dry_run
```

Learned routing uses a fixed 80/20 training/validation split. Model family, feature set, and threshold are selected on validation data before a single test-set application. See `docs/selectors.md` for required score-file layout.

## 7. External benchmarks

GenEval and T2I-CompBench adapters are included, but their official datasets, detector weights, and evaluator environments are not redistributed. Follow `docs/benchmarks.md` and the upstream projects. This separation avoids silently substituting unofficial evaluators.

## 8. Training

- Prompt-editor LoRA SFT: `scripts/train_intent_preserving_lora_sft.py`
- CoMat: `scripts/train_comat.py`

Training data and learned weights are not included unless explicitly listed in `MANIFEST.md`.

## Determinism notes

- Prompt refinement uses the entry-point defaults recorded in code.
- The learned selector default seed is 995 and the split is exactly 80/20.
- LoRA SFT default seed is 42.
- Image generation uses integer sample seeds beginning at zero unless a benchmark adapter receives a seed offset.
- GPU kernels, model revisions, API providers, and external evaluator versions can still introduce nondeterminism. Record the resolved model revision, hardware, driver, PyTorch version, and command line when reproducing tables.

