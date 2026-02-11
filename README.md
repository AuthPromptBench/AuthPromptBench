# APBench Toolkit

APBench Toolkit is a research codebase for intent-preserving text-to-image prompt editing, multi-model image generation, evaluation, and prompt routing.

This distribution is prepared as an anonymous paper Code and Data Supplement. It contains no author names, affiliations, personal paths, credentials, checkpoints, generated images, or private experiment logs.

## Included functionality

- Prompt editing with Promptist, PAG, PAE, local Llama/Mistral/Qwen models, LoRA checkpoints, and API models.
- Image generation with SD2.1, SD3, FLUX.1, PixArt-Sigma, CoMat, Infinity, and Show-O2.
- APBench CLIP and Qwen-VL evaluation plus result aggregation.
- GenEval and T2I-CompBench generation/evaluation adapters.
- Zero-shot LLM routing and per-generator learned routers using a fixed 80/20 train/validation split followed by one test-set evaluation.
- LoRA SFT prompt-editor training and CoMat training.

Historical experiment sweeps, private cluster paths, generated outputs, model weights, and unrelated training objectives are intentionally excluded.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CUDA or ROCm environment separately. Some generators also require their upstream repositories and checkpoints; see [docs/generation.md](docs/generation.md).

## Quick start

Edit prompts:

```bash
python metrics/refine.py --refiner_type qwen3
python metrics/refine.py --refiner_type llama3_lora --checkpoint checkpoints/refiner.pt
python metrics/refine.py --refiner_type gpt-4o-mini
```

Generate and evaluate:

```bash
python metrics/generate.py --model sd21 --refiner_type qwen3
python -m metrics.eval \
  --images_dir data/benchmark/generated_images/qwen3/sd21 \
  --result_dir data/benchmark/results/qwen3/sd21
python metrics/print_res.py data/benchmark/results/qwen3/sd21
```

See [docs/refinement-and-training.md](docs/refinement-and-training.md), [docs/generation.md](docs/generation.md), [docs/benchmarks.md](docs/benchmarks.md), and [docs/selectors.md](docs/selectors.md).

For paper reproduction, start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Dataset fields, provenance, and integrity information are in [DATA_CARD.md](DATA_CARD.md). The intended submission contents are listed in [MANIFEST.md](MANIFEST.md).

## Data and checkpoints

Large weights and generated artifacts are not versioned. Put private or gated credentials in environment variables. Before redistributing benchmark data or third-party checkpoints, verify their upstream licenses and terms.

## License

The repository code is released under the MIT License. Third-party models, datasets, and evaluator repositories retain their own licenses.
