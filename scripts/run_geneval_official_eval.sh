#!/bin/bash
# Run the official GenEval evaluator once its CUDA/mmdet environment is active.
#
# Required external setup:
#   git clone https://github.com/djghosh13/geneval.git <GENEVAL_REPO>
#   conda env create -f <GENEVAL_REPO>/environment.yml
#   conda activate geneval
#   <GENEVAL_REPO>/evaluation/download_models.sh <DETECTOR_DIR>
#   git clone https://github.com/open-mmlab/mmdetection.git
#   cd mmdetection && git checkout 2.x && pip install -v -e .

set -euo pipefail

: "${GENEVAL_REPO:?Set GENEVAL_REPO to the official geneval checkout.}"
: "${IMAGE_DIR:?Set IMAGE_DIR to a GenEval-format image directory.}"
: "${RESULT_DIR:?Set RESULT_DIR for results.jsonl and summary.txt.}"
: "${DETECTOR_DIR:?Set DETECTOR_DIR containing mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth.}"

mkdir -p "${RESULT_DIR}"

python "${GENEVAL_REPO}/evaluation/evaluate_images.py" \
  "${IMAGE_DIR}" \
  --outfile "${RESULT_DIR}/results.jsonl" \
  --model-path "${DETECTOR_DIR}"

python "${GENEVAL_REPO}/evaluation/summary_scores.py" \
  "${RESULT_DIR}/results.jsonl" | tee "${RESULT_DIR}/summary.txt"
