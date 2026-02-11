# External benchmarks

## GenEval

Fetch the official GenEval repository and metadata according to its upstream license. Generate images in the official directory layout with:

```bash
python scripts/geneval_generate_sharded.py --help
```

Run the official detector environment through the generic wrapper:

```bash
GENEVAL_REPO=/path/to/geneval \
IMAGE_DIR=/path/to/images \
RESULT_DIR=/path/to/results \
DETECTOR_DIR=/path/to/detector \
bash scripts/run_geneval_official_eval.sh
```

## T2I-CompBench

```bash
python scripts/fetch_t2i_compbench_dataset.py --help
python scripts/build_t2i_compbench_combined_val_prompts.py --name all_val
python scripts/t2i_compbench_generate_sharded.py --help
python scripts/materialize_t2i_compbench_category_views.py --help
python scripts/summarize_t2i_compbench_results.py --help
```

The official evaluator and its weights are not vendored. Install them in a separate environment and follow the upstream repository's license and setup instructions.

