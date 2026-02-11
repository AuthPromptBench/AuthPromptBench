# Image generation and APBench evaluation

Supported generator names are `sd21`, `sd3`, `flux1`, `pixart-sigma`, `comat21`, `infi`, and `showo2`. `pae_sd21` is also available for PAE generation.

```bash
python metrics/generate.py --model flux1
python metrics/generate.py --model flux1 --refiner_type qwen3
python metrics/generate.py --model sd21 --start_index 0 --end_index 32 --images_per_prompt 4
```

SD, FLUX, and PixArt weights are resolved through Hugging Face. CoMat expects the checkpoint configured by `COMAT21_CHECKPOINT`; Infinity and Show-O2 require their upstream code/configuration paths through the environment variables documented in `metrics/generate.py`.

APBench evaluation writes resumable CLIP and Qwen JSONL files:

```bash
python -m metrics.eval --images_dir PATH --result_dir PATH
python metrics/print_res.py PATH
```

The evaluator requires a CUDA-capable environment for the current CLIP/Qwen implementations.

