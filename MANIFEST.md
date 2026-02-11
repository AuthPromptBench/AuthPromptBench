# Supplement manifest

The submitted archive is intended to contain only the following categories.

| Path | Purpose |
|---|---|
| `README.md` | Entry point and scope |
| `REPRODUCIBILITY.md` | End-to-end reproduction guide |
| `DATA_CARD.md` | Dataset schema, provenance, integrity, and limitations |
| `MANIFEST.md` | Submission contents |
| `LICENSE` | License for repository code |
| `THIRD_PARTY_NOTICES.md` | Upstream components and separate-license notice |
| `.env.example` | Names of optional environment variables; no values |
| `.gitignore` | Exclusion rules for generated/private artifacts |
| `requirements.txt` | Pinned direct Python dependencies |
| `configs/` | APBench class names and CoMat model configurations |
| `data/benchmark/auth_prompt_V1_0_0.json` | Included APBench metadata, 2,048 records |
| `data/benchmark/refined_prompts/.gitkeep` | Empty output-directory marker |
| `docs/` | Component-specific usage notes |
| `metrics/` | Prompt editing, image generation, APBench evaluation, result aggregation |
| `models/` | CoMat, prompt editor, PAE, and supporting model components |
| `scripts/` | Training, routing, GenEval, and T2I-CompBench entry points |

The archive must not contain `.git/`, `.env`, caches, logs, checkpoints, generated images, evaluation outputs, API responses, scheduler scripts, private paths, or editor metadata.

Before submission, compare the actual file tree with this manifest and run the scans documented in `REPRODUCIBILITY.md`.
