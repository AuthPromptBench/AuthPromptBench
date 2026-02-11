#!/usr/bin/env python3
"""Train a selector on non-APBench qwen labels and test it on APBench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GENERATORS = ("comat21", "showo2", "sd21", "flux1", "sd3", "infi", "pixart-sigma")


def read_json(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def qwen_mean(row: dict) -> float:
    scores = row.get("qwen_sentence_scores") or []
    if not scores:
        return 0.0
    return float(mean(float(x) for x in scores))


def load_scores(root: Path, result_root: str, methods: tuple[str, str]) -> dict[str, dict[str, dict[str, float]]]:
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for generator in GENERATORS:
        scores[generator] = {}
        for method in methods:
            path = root / result_root / method / generator / "qwen.jsonl"
            rows = read_jsonl(path)
            scores[generator][method] = {str(row["id"]): qwen_mean(row) for row in rows}
    return scores


def mean_pool(last_hidden_state, attention_mask):
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    torch_dtype: str,
    trust_remote_code: bool,
) -> np.ndarray:
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype_map[torch_dtype],
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc=f"Embedding {model_name}"):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            outputs = model(**inputs)
            pooled = mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def build_features(original_emb: np.ndarray, rewrite_emb: np.ndarray) -> dict[str, np.ndarray]:
    diff = rewrite_emb - original_emb
    absdiff = np.abs(diff)
    product = rewrite_emb * original_emb
    cosine = (rewrite_emb * original_emb).sum(axis=1, keepdims=True)
    return {
        "original_only": original_emb,
        "rewrite_only": rewrite_emb,
        "concat": np.concatenate([original_emb, rewrite_emb], axis=1),
        "diff_abs": np.concatenate([diff, absdiff, cosine], axis=1),
        "pair_full": np.concatenate([original_emb, rewrite_emb, diff, absdiff, product, cosine], axis=1),
    }


def make_classifier(name: str):
    if name == "logreg_c1":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=995)),
            ]
        )
    if name == "logreg_c100":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, C=100.0, class_weight="balanced", random_state=995)),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=995, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=500, class_weight="balanced", random_state=995, n_jobs=-1)
    raise ValueError(name)


def fixed_mean(ids: list[str], generator: str, method: str, scores: dict) -> float:
    return float(np.mean([scores[generator][method][sample_id] for sample_id in ids]))


def oracle_mean(ids: list[str], generator: str, methods: tuple[str, str], scores: dict) -> float:
    a, b = methods
    return float(np.mean([max(scores[generator][a][sample_id], scores[generator][b][sample_id]) for sample_id in ids]))


def routed_mean(ids: list[str], pred_rewrite: np.ndarray, generator: str, methods: tuple[str, str], scores: dict) -> float:
    default_method, rewrite_method = methods
    values = []
    for sample_id, choose_rewrite in zip(ids, pred_rewrite.astype(bool)):
        method = rewrite_method if choose_rewrite else default_method
        values.append(scores[generator][method][sample_id])
    return float(np.mean(values))


