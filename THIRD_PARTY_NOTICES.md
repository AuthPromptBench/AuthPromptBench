# Third-party notices

This supplement interfaces with or contains adapted implementation patterns from third-party projects. Their names and links are provided for attribution and reproducibility; their code, models, datasets, and weights remain subject to their respective licenses and terms.

- Hugging Face Transformers and Diffusers: model loading and diffusion pipeline foundations.
- CoMat-related training components: compositional text-to-image training implementation.
- Promptist, PAG, and PAE: traditional prompt-refinement methods; checkpoints are not included.
- TokenCompose and Prompt-to-Prompt: attention-control utilities referenced in source comments.
- BLIP, Grounding DINO, FastSAM/Ultralytics: CoMat reward and segmentation components.
- GenEval: official prompt metadata/evaluator integration; the upstream evaluator is not vendored.
- T2I-CompBench: official prompt/evaluator integration; the upstream evaluator is not vendored.
- Infinity and Show-O2: optional external image generators; upstream repositories and weights are not vendored.
- Stable Diffusion, FLUX.1, PixArt-Sigma, Llama, Mistral, Qwen, and API-hosted models: optional model dependencies governed by their model cards and provider terms.

Source files containing more specific upstream references retain those references in comments. The top-level MIT license applies only to material for which the supplement authors have authority to grant that license. Confirm compatibility of all adapted components and benchmark metadata before a public, non-review release.
