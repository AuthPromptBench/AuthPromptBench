#!/usr/bin/env python3
"""Train gain regressors on non-APBench and route APBench prompts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from difflib import SequenceMatcher
from statistics import mean

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from selector_utils import (
    GENERATORS,
    embed_texts,
    fixed_mean,
    load_scores,
    oracle_mean,
    read_json,
    routed_mean,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
SENTENCE_RE = re.compile(r"[.!?。！？]+")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        short, long = a, b
    else:
        short, long = b, a
    prev = [0] * (len(short) + 1)
    for token in long:
        cur = [0]
        for j, other in enumerate(short, start=1):
            cur.append(prev[j - 1] + 1 if token == other else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def sentence_count(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    parts = [part for part in SENTENCE_RE.split(stripped) if part.strip()]
    return max(1, len(parts))


def surface_features(original_prompts: list[str], rewrite_prompts: list[str]) -> np.ndarray:
    rows = []
    for original, rewrite in zip(original_prompts, rewrite_prompts):
        original_tokens = tokenize(original)
        rewrite_tokens = tokenize(rewrite)
        original_set = set(original_tokens)
        rewrite_set = set(rewrite_tokens)
        inter = original_set & rewrite_set
        union = original_set | rewrite_set
        original_token_count = len(original_tokens)
        rewrite_token_count = len(rewrite_tokens)
        original_char_count = len(original)
        rewrite_char_count = len(rewrite)
        lcs = lcs_len(original_tokens, rewrite_tokens)
        rows.append(
            [
                original_token_count,
                rewrite_token_count,
                rewrite_token_count / max(original_token_count, 1),
                rewrite_token_count - original_token_count,
                rewrite_char_count / max(original_char_count, 1),
                sentence_count(rewrite) - sentence_count(original),
                len(inter) / max(len(union), 1),
                len(inter) / max(len(original_set), 1),
                len(inter) / max(len(rewrite_set), 1),
                1.0 - SequenceMatcher(None, original.lower(), rewrite.lower()).ratio(),
                lcs / max(max(original_token_count, rewrite_token_count), 1),
                len(rewrite_set - original_set) / max(len(rewrite_set), 1),
                len(original_set - rewrite_set) / max(len(original_set), 1),
            ]
        )
    return np.asarray(rows, dtype=float)


def build_enhanced_features(
    train_original_emb: np.ndarray,
    train_rewrite_emb: np.ndarray,
    test_original_emb: np.ndarray,
    test_rewrite_emb: np.ndarray,
    train_surface: np.ndarray,
    test_surface: np.ndarray,
    pca_components: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train_diff = train_rewrite_emb - train_original_emb
    test_diff = test_rewrite_emb - test_original_emb
    train_absdiff = np.abs(train_diff)
    test_absdiff = np.abs(test_diff)
    train_product = train_rewrite_emb * train_original_emb
    test_product = test_rewrite_emb * test_original_emb
    train_cosine = (train_rewrite_emb * train_original_emb).sum(axis=1, keepdims=True)
    test_cosine = (test_rewrite_emb * test_original_emb).sum(axis=1, keepdims=True)

    train_pair_full = np.concatenate(
        [train_original_emb, train_rewrite_emb, train_diff, train_absdiff, train_product, train_cosine],
        axis=1,
    )
    test_pair_full = np.concatenate(
        [test_original_emb, test_rewrite_emb, test_diff, test_absdiff, test_product, test_cosine],
        axis=1,
    )
    n_components = min(pca_components, train_pair_full.shape[0], train_pair_full.shape[1])
    pca = PCA(n_components=n_components, random_state=995)
    train_pair_pca = pca.fit_transform(train_pair_full)
    test_pair_pca = pca.transform(test_pair_full)

    train_base = {
        "surface_only": train_surface,
        "original_surface": np.concatenate([train_original_emb, train_surface], axis=1),
        "rewrite_surface": np.concatenate([train_rewrite_emb, train_surface], axis=1),
        "concat_surface": np.concatenate([train_original_emb, train_rewrite_emb, train_surface], axis=1),
        "diff_abs_surface": np.concatenate([train_diff, train_absdiff, train_cosine, train_surface], axis=1),
        "pair_full_pca_surface": np.concatenate([train_pair_pca, train_surface], axis=1),
    }
    test_base = {
        "surface_only": test_surface,
        "original_surface": np.concatenate([test_original_emb, test_surface], axis=1),
        "rewrite_surface": np.concatenate([test_rewrite_emb, test_surface], axis=1),
        "concat_surface": np.concatenate([test_original_emb, test_rewrite_emb, test_surface], axis=1),
        "diff_abs_surface": np.concatenate([test_diff, test_absdiff, test_cosine, test_surface], axis=1),
        "pair_full_pca_surface": np.concatenate([test_pair_pca, test_surface], axis=1),
    }
    return train_base, test_base


def pair_pca_features(
    train_original_emb: np.ndarray,
    train_rewrite_emb: np.ndarray,
    test_original_emb: np.ndarray,
    test_rewrite_emb: np.ndarray,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_diff = train_rewrite_emb - train_original_emb
    test_diff = test_rewrite_emb - test_original_emb
    train_absdiff = np.abs(train_diff)
    test_absdiff = np.abs(test_diff)
    train_product = train_rewrite_emb * train_original_emb
    test_product = test_rewrite_emb * test_original_emb
    train_cosine = (train_rewrite_emb * train_original_emb).sum(axis=1, keepdims=True)
    test_cosine = (test_rewrite_emb * test_original_emb).sum(axis=1, keepdims=True)
    train_pair_full = np.concatenate(
        [train_original_emb, train_rewrite_emb, train_diff, train_absdiff, train_product, train_cosine],
        axis=1,
    )
    test_pair_full = np.concatenate(
        [test_original_emb, test_rewrite_emb, test_diff, test_absdiff, test_product, test_cosine],
        axis=1,
    )
    n_components = min(pca_components, train_pair_full.shape[0], train_pair_full.shape[1])
    pca = PCA(n_components=n_components, random_state=995)
    return pca.fit_transform(train_pair_full), pca.transform(test_pair_full), train_diff, test_diff


def target_t2i_embedding_path(root: Path, embedding_dir: Path, generator: str, train_rewrite: str, test_rewrite: str) -> Path:
    safe_generator = generator.replace("/", "__")
    return root / embedding_dir / (
        f"{safe_generator}_target_text_encoder_train_{train_rewrite}_test_{test_rewrite}.npz"
    )


def build_target_t2i_features(
    root: Path,
    embedding_dir: Path,
    generator: str,
    train_rewrite: str,
    test_rewrite: str,
    train_ids: list[str],
    test_ids: list[str],
    train_surface: np.ndarray,
    test_surface: np.ndarray,
    qwen_train_pair_surface: np.ndarray,
    qwen_test_pair_surface: np.ndarray,
    pca_components: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    path = target_t2i_embedding_path(root, embedding_dir, generator, train_rewrite, test_rewrite)
    if not path.exists():
        raise FileNotFoundError(f"Missing target T2I embedding file: {path}")
    data = np.load(path, allow_pickle=True)
    if list(map(str, data["train_ids"].tolist())) != train_ids:
        raise ValueError(f"Train ids do not match target T2I embeddings for {generator}.")
    if list(map(str, data["test_ids"].tolist())) != test_ids:
        raise ValueError(f"Test ids do not match target T2I embeddings for {generator}.")

    train_original = data["train_original_emb"].astype(float)
    train_rewrite_emb = data["train_rewrite_emb"].astype(float)
    test_original = data["test_original_emb"].astype(float)
    test_rewrite_emb = data["test_rewrite_emb"].astype(float)
    train_pair_pca, test_pair_pca, train_diff, test_diff = pair_pca_features(
        train_original,
        train_rewrite_emb,
        test_original,
        test_rewrite_emb,
        pca_components,
    )
    train_scalars = np.stack(
        [
            data["train_target_cosine"].astype(float),
            data["train_target_embedding_shift_norm"].astype(float),
            data["train_target_original_norm"].astype(float),
            data["train_target_rewrite_norm"].astype(float),
        ],
        axis=1,
    )
    test_scalars = np.stack(
        [
            data["test_target_cosine"].astype(float),
            data["test_target_embedding_shift_norm"].astype(float),
            data["test_target_original_norm"].astype(float),
            data["test_target_rewrite_norm"].astype(float),
        ],
        axis=1,
    )
    train_absdiff = np.abs(train_diff)
    test_absdiff = np.abs(test_diff)
    train_features = {
        "target_t2i_scalar_only": train_scalars,
        "target_t2i_pair_pca_only": np.concatenate([train_pair_pca, train_scalars], axis=1),
        "target_t2i_diff_abs_only": np.concatenate([train_diff, train_absdiff, train_scalars], axis=1),
        "target_t2i_scalar_surface": np.concatenate([train_scalars, train_surface], axis=1),
        "target_t2i_pair_pca_surface": np.concatenate([train_pair_pca, train_scalars, train_surface], axis=1),
        "target_t2i_diff_abs_surface": np.concatenate([train_diff, train_absdiff, train_scalars, train_surface], axis=1),
        "qwen_target_pair_pca_surface": np.concatenate([qwen_train_pair_surface, train_pair_pca, train_scalars], axis=1),
    }
    test_features = {
        "target_t2i_scalar_only": test_scalars,
        "target_t2i_pair_pca_only": np.concatenate([test_pair_pca, test_scalars], axis=1),
        "target_t2i_diff_abs_only": np.concatenate([test_diff, test_absdiff, test_scalars], axis=1),
        "target_t2i_scalar_surface": np.concatenate([test_scalars, test_surface], axis=1),
        "target_t2i_pair_pca_surface": np.concatenate([test_pair_pca, test_scalars, test_surface], axis=1),
        "target_t2i_diff_abs_surface": np.concatenate([test_diff, test_absdiff, test_scalars, test_surface], axis=1),
        "qwen_target_pair_pca_surface": np.concatenate([qwen_test_pair_surface, test_pair_pca, test_scalars], axis=1),
    }
    return train_features, test_features


def build_tfidf_features(
    train_original_prompts: list[str],
    train_rewrite_prompts: list[str],
    test_original_prompts: list[str],
    test_rewrite_prompts: list[str],
    max_features: int,
) -> tuple[dict[str, object], dict[str, object]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[\w][\w'-]+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=max_features,
    )
    vectorizer.fit(train_original_prompts + train_rewrite_prompts)
    train_original = vectorizer.transform(train_original_prompts)
    train_rewrite = vectorizer.transform(train_rewrite_prompts)
    test_original = vectorizer.transform(test_original_prompts)
    test_rewrite = vectorizer.transform(test_rewrite_prompts)
    return (
        {
            "tfidf_concat": hstack([train_original, train_rewrite], format="csr"),
            "tfidf_diff_abs": hstack([train_rewrite - train_original, abs(train_rewrite - train_original)], format="csr"),
        },
        {
            "tfidf_concat": hstack([test_original, test_rewrite], format="csr"),
            "tfidf_diff_abs": hstack([test_rewrite - test_original, abs(test_rewrite - test_original)], format="csr"),
        },
    )


def make_regressor(name: str):
    if name == "ridge_a1":
        return Pipeline([("scale", StandardScaler(with_mean=False)), ("reg", Ridge(alpha=1.0))])
    if name == "ridge_a10":
        return Pipeline([("scale", StandardScaler(with_mean=False)), ("reg", Ridge(alpha=10.0))])
    if name == "elasticnet_a001_l1_05":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("reg", ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=995)),
            ]
        )
    if name == "elasticnet_a01_l1_02":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("reg", ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=995)),
            ]
        )
    if name == "linear_svr":
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=False)),
                ("reg", LinearSVR(C=1.0, epsilon=0.01, max_iter=10000, random_state=995)),
            ]
        )
    if name == "hist_gbr":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, l2_regularization=0.01, random_state=995)
    if name == "random_forest_reg":
        return RandomForestRegressor(n_estimators=200, random_state=995, n_jobs=-1, min_samples_leaf=2)
    if name == "extra_trees_reg":
        return ExtraTreesRegressor(n_estimators=200, random_state=995, n_jobs=-1, min_samples_leaf=2)
    if name == "gbr":
        return GradientBoostingRegressor(random_state=995, n_estimators=300, learning_rate=0.03, max_depth=3)
    if name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=500,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=995,
            n_jobs=-1,
        )
    if name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03, loss_function="RMSE", random_seed=995, verbose=False)
    raise ValueError(name)


def threshold_sweep(
    ids: list[str],
    pred_gain: np.ndarray,
    generator: str,
    methods: tuple[str, str],
    scores: dict,
) -> tuple[float, int, float]:
    default_method, rewrite_method = methods
    default_values = np.array([scores[generator][default_method][sample_id] for sample_id in ids], dtype=float)
    rewrite_values = np.array([scores[generator][rewrite_method][sample_id] for sample_id in ids], dtype=float)
    current_sum = float(default_values.sum())
    best_score = current_sum / len(ids)
    best_threshold = float("inf")
    best_selected = 0

    order = np.argsort(-pred_gain)
    selected = 0
    start = 0
    while start < len(order):
        threshold = float(pred_gain[order[start]])
        end = start
        while end < len(order) and pred_gain[order[end]] == threshold:
            idx = order[end]
            current_sum += float(rewrite_values[idx] - default_values[idx])
            selected += 1
            end += 1
        routed = current_sum / len(ids)
        if routed > best_score:
            best_score = routed
            best_threshold = threshold
            best_selected = selected
        start = end
    return best_threshold, best_selected, best_score


def safe_corr(fn, a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(fn(a, b).statistic)


