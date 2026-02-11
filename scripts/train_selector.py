#!/usr/bin/env python3
"""Train per-generator prompt routers with an 80/20 split and evaluate once on a test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from selector_features import (
    GENERATORS,
    build_enhanced_features,
    build_tfidf_features,
    embed_texts,
    make_regressor,
    safe_name,
    safe_corr,
    surface_features,
    target_t2i_embedding_path,
)
from selector_utils import fixed_mean, load_scores, oracle_mean, read_json, routed_mean


def take_rows(values, indices: np.ndarray):
    return values[indices] if not hasattr(values, "tocsr") else values.tocsr()[indices]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def threshold_sweep_arrays(default_values: np.ndarray, rewrite_values: np.ndarray, pred_gain: np.ndarray) -> tuple[float, int, float]:
    current_sum = float(default_values.sum())
    best_score = current_sum / len(default_values)
    best_threshold = float("inf")
    best_selected = 0
    order = np.argsort(-pred_gain)
    selected = 0
    start = 0
    while start < len(order):
        threshold = float(pred_gain[order[start]])
        end = start
        while end < len(order) and pred_gain[order[end]] == threshold:
            idx = int(order[end])
            current_sum += float(rewrite_values[idx] - default_values[idx])
            selected += 1
            end += 1
        routed = current_sum / len(default_values)
        if routed > best_score:
            best_score = routed
            best_threshold = threshold
            best_selected = selected
        start = end
    return best_threshold, best_selected, best_score


def route_with_threshold(ids: list[str], pred_gain: np.ndarray, threshold: float, generator: str, methods: tuple[str, str], scores: dict) -> tuple[float, int]:
    if np.isinf(threshold):
        choose_rewrite = np.zeros(len(ids), dtype=bool)
    else:
        choose_rewrite = pred_gain >= threshold
    return routed_mean(ids, choose_rewrite, generator, methods, scores), int(choose_rewrite.sum())


def score_vectors(ids: list[str], generator: str, default_method: str, rewrite_method: str, scores: dict) -> tuple[np.ndarray, np.ndarray]:
    default_values = np.array([scores[generator][default_method][sample_id] for sample_id in ids], dtype=float)
    rewrite_values = np.array([scores[generator][rewrite_method][sample_id] for sample_id in ids], dtype=float)
    return default_values, rewrite_values


def subset_lists(values: list[str], indices: np.ndarray) -> list[str]:
    return [values[int(i)] for i in indices]


def build_base_feature_sets(
    original_emb: np.ndarray,
    rewrite_emb: np.ndarray,
    original_prompts: list[str],
    rewrite_prompts: list[str],
    train_idx: np.ndarray,
    eval_idx: np.ndarray | None,
    eval_original_emb: np.ndarray | None = None,
    eval_rewrite_emb: np.ndarray | None = None,
    eval_original_prompts: list[str] | None = None,
    eval_rewrite_prompts: list[str] | None = None,
    pca_components: int = 128,
    tfidf_max_features: int = 4096,
) -> tuple[dict[str, object], dict[str, object]]:
    train_original_emb = original_emb[train_idx]
    train_rewrite_emb = rewrite_emb[train_idx]
    train_original_prompts = subset_lists(original_prompts, train_idx)
    train_rewrite_prompts = subset_lists(rewrite_prompts, train_idx)

    if eval_idx is not None:
        eval_original_emb2 = original_emb[eval_idx]
        eval_rewrite_emb2 = rewrite_emb[eval_idx]
        eval_original_prompts2 = subset_lists(original_prompts, eval_idx)
        eval_rewrite_prompts2 = subset_lists(rewrite_prompts, eval_idx)
    else:
        assert eval_original_emb is not None and eval_rewrite_emb is not None
        assert eval_original_prompts is not None and eval_rewrite_prompts is not None
        eval_original_emb2 = eval_original_emb
        eval_rewrite_emb2 = eval_rewrite_emb
        eval_original_prompts2 = eval_original_prompts
        eval_rewrite_prompts2 = eval_rewrite_prompts

    train_surface = surface_features(train_original_prompts, train_rewrite_prompts)
    eval_surface = surface_features(eval_original_prompts2, eval_rewrite_prompts2)
    train_features, eval_features = build_enhanced_features(
        train_original_emb,
        train_rewrite_emb,
        eval_original_emb2,
        eval_rewrite_emb2,
        train_surface,
        eval_surface,
        pca_components,
    )
    train_tfidf, eval_tfidf = build_tfidf_features(
        train_original_prompts,
        train_rewrite_prompts,
        eval_original_prompts2,
        eval_rewrite_prompts2,
        tfidf_max_features,
    )
    train_features.update(train_tfidf)
    eval_features.update(eval_tfidf)
    return train_features, eval_features


def add_target_features_if_available(
    root: Path,
    embedding_dir: Path,
    generator: str,
    train_rewrite_name: str,
    test_rewrite_name: str,
    train_ids_full: list[str],
    test_ids: list[str],
    train_idx: np.ndarray,
    eval_idx: np.ndarray | None,
    train_features: dict[str, object],
    eval_features: dict[str, object],
    pca_components: int,
) -> None:
    from sklearn.decomposition import PCA

    path = target_t2i_embedding_path(root, embedding_dir, generator, train_rewrite_name, test_rewrite_name)
    if not path.exists():
        print(f"[info] target T2I embeddings missing for {generator}: {path}", flush=True)
        return
    data = np.load(path, allow_pickle=True)
    if list(map(str, data["train_ids"].tolist())) != train_ids_full:
        raise ValueError(f"Train ids do not match target T2I embeddings for {generator}.")
    if eval_idx is None and list(map(str, data["test_ids"].tolist())) != test_ids:
        raise ValueError(f"Test ids do not match target T2I embeddings for {generator}.")

    train_original = data["train_original_emb"].astype(float)[train_idx]
    train_rewrite = data["train_rewrite_emb"].astype(float)[train_idx]
    if eval_idx is None:
        eval_original = data["test_original_emb"].astype(float)
        eval_rewrite = data["test_rewrite_emb"].astype(float)
        eval_scalars = np.stack(
            [
                data["test_target_cosine"].astype(float),
                data["test_target_embedding_shift_norm"].astype(float),
                data["test_target_original_norm"].astype(float),
                data["test_target_rewrite_norm"].astype(float),
            ],
            axis=1,
        )
    else:
        eval_original = data["train_original_emb"].astype(float)[eval_idx]
        eval_rewrite = data["train_rewrite_emb"].astype(float)[eval_idx]
        eval_scalars = np.stack(
            [
                data["train_target_cosine"].astype(float)[eval_idx],
                data["train_target_embedding_shift_norm"].astype(float)[eval_idx],
                data["train_target_original_norm"].astype(float)[eval_idx],
                data["train_target_rewrite_norm"].astype(float)[eval_idx],
            ],
            axis=1,
        )
    train_scalars = np.stack(
        [
            data["train_target_cosine"].astype(float)[train_idx],
            data["train_target_embedding_shift_norm"].astype(float)[train_idx],
            data["train_target_original_norm"].astype(float)[train_idx],
            data["train_target_rewrite_norm"].astype(float)[train_idx],
        ],
        axis=1,
    )
    train_diff = train_rewrite - train_original
    eval_diff = eval_rewrite - eval_original
    train_absdiff = np.abs(train_diff)
    eval_absdiff = np.abs(eval_diff)
    train_pair = np.concatenate([train_original, train_rewrite, train_diff, train_absdiff, train_original * train_rewrite], axis=1)
    eval_pair = np.concatenate([eval_original, eval_rewrite, eval_diff, eval_absdiff, eval_original * eval_rewrite], axis=1)
    n_components = min(pca_components, train_pair.shape[0], train_pair.shape[1])
    pca = PCA(n_components=n_components, random_state=995)
    train_pair_pca = pca.fit_transform(train_pair)
    eval_pair_pca = pca.transform(eval_pair)
    train_features.update(
        {
            "target_t2i_scalar_only": train_scalars,
            "target_t2i_pair_pca_only": np.concatenate([train_pair_pca, train_scalars], axis=1),
            "target_t2i_diff_abs_only": np.concatenate([train_diff, train_absdiff, train_scalars], axis=1),
        }
    )
    eval_features.update(
        {
            "target_t2i_scalar_only": eval_scalars,
            "target_t2i_pair_pca_only": np.concatenate([eval_pair_pca, eval_scalars], axis=1),
            "target_t2i_diff_abs_only": np.concatenate([eval_diff, eval_absdiff, eval_scalars], axis=1),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train_benchmark", type=Path, required=True)
    parser.add_argument("--test_benchmark", type=Path, required=True)
    parser.add_argument("--train_default", required=True)
    parser.add_argument("--train_rewrite", required=True)
    parser.add_argument("--test_default", required=True)
    parser.add_argument("--test_rewrite", required=True)
    parser.add_argument("--result_root", default="data/benchmark/results")
    parser.add_argument("--model_name", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/selectors"))
    parser.add_argument("--output_tag", default="", help="Optional suffix inserted into output filenames.")
    parser.add_argument("--generators", nargs="+", default=list(GENERATORS), choices=GENERATORS)
    parser.add_argument("--validation_fraction", type=float, default=0.2, choices=[0.2])
    parser.add_argument("--seed", type=int, default=995)
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--target_pca_components", type=int, default=128)
    parser.add_argument("--tfidf_max_features", type=int, default=4096)
    parser.add_argument("--include_target_t2i_embeddings", action="store_true")
    parser.add_argument("--target_t2i_embedding_dir", type=Path, default=Path("data/benchmark/analysis/target_t2i_text_encoder_embeddings"))
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=["ridge_a1", "ridge_a10", "elasticnet_a001_l1_05", "elasticnet_a01_l1_02", "hist_gbr", "xgboost"],
    )
    args = parser.parse_args()

    root = args.root
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_json(root / args.train_benchmark)
    test_rows = read_json(root / args.test_benchmark)
    train_rewrite_rows = read_json(root / "data/benchmark/refined_prompts" / f"{args.train_rewrite}.json")
    test_rewrite_rows = read_json(root / "data/benchmark/refined_prompts" / f"{args.test_rewrite}.json")
    train_ids = [str(row["id"]) for row in train_rows]
    test_ids = [str(row["id"]) for row in test_rows]
    if train_ids != [str(row["id"]) for row in train_rewrite_rows]:
        raise ValueError("Train benchmark and rewrite prompt orders do not match.")
    if test_ids != [str(row["id"]) for row in test_rewrite_rows]:
        raise ValueError("APBench and rewrite prompt orders do not match.")

    train_scores = load_scores(root, args.result_root, (args.train_default, args.train_rewrite))
    test_scores = load_scores(root, args.result_root, (args.test_default, args.test_rewrite))

    model_safe = args.model_name.replace("/", "__")
    train_stem = safe_name(args.train_benchmark.stem)
    embedding_generator_suffix = (
        "all" if tuple(args.generators) == tuple(GENERATORS) else "_".join(args.generators).replace("-", "_")
    )
    embedding_path = out_dir / (
        f"{model_safe}_{embedding_generator_suffix}_{train_stem}_train_{safe_name(args.train_rewrite)}"
        f"_test_{safe_name(args.test_rewrite)}_embeddings.npz"
    )
    train_original_prompts = [row.get("prompt") or "" for row in train_rows]
    train_rewrite_prompts = [row.get("refined_prompt") or row.get("prompt") or "" for row in train_rewrite_rows]
    test_original_prompts = [row.get("prompt") or "" for row in test_rows]
    test_rewrite_prompts = [row.get("refined_prompt") or row.get("prompt") or "" for row in test_rewrite_rows]
    if embedding_path.exists():
        emb = np.load(embedding_path, allow_pickle=True)
        train_original_emb = emb["train_original_emb"].astype(float)
        train_rewrite_emb = emb["train_rewrite_emb"].astype(float)
        test_original_emb = emb["test_original_emb"].astype(float)
        test_rewrite_emb = emb["test_rewrite_emb"].astype(float)
    else:
        unique_texts = list(dict.fromkeys(train_original_prompts + train_rewrite_prompts + test_original_prompts + test_rewrite_prompts))
        embeddings = embed_texts(
            unique_texts,
            model_name=args.model_name,
            batch_size=args.batch_size,
            max_length=args.max_length,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
        emb_by_text = {text: embeddings[i] for i, text in enumerate(unique_texts)}
        train_original_emb = np.stack([emb_by_text[text] for text in train_original_prompts])
        train_rewrite_emb = np.stack([emb_by_text[text] for text in train_rewrite_prompts])
        test_original_emb = np.stack([emb_by_text[text] for text in test_original_prompts])
        test_rewrite_emb = np.stack([emb_by_text[text] for text in test_rewrite_prompts])
        np.savez_compressed(
            embedding_path,
            train_original_emb=train_original_emb,
            train_rewrite_emb=train_rewrite_emb,
            test_original_emb=test_original_emb,
            test_rewrite_emb=test_rewrite_emb,
            train_ids=np.array(train_ids, dtype=object),
            test_ids=np.array(test_ids, dtype=object),
        )

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_ids))
    val_size = int(round(len(train_ids) * args.validation_fraction))
    val_idx = np.sort(perm[:val_size])
    fit_idx = np.sort(perm[val_size:])
    fit_ids = [train_ids[int(i)] for i in fit_idx]
    val_ids = [train_ids[int(i)] for i in val_idx]
    print(f"fit={len(fit_idx)} val={len(val_idx)} test={len(test_ids)}", flush=True)

    val_train_features, val_features = build_base_feature_sets(
        train_original_emb,
        train_rewrite_emb,
        train_original_prompts,
        train_rewrite_prompts,
        fit_idx,
        val_idx,
        pca_components=args.pca_components,
        tfidf_max_features=args.tfidf_max_features,
    )
    test_train_features, test_features = build_base_feature_sets(
        train_original_emb,
        train_rewrite_emb,
        train_original_prompts,
        train_rewrite_prompts,
        fit_idx,
        None,
        eval_original_emb=test_original_emb,
        eval_rewrite_emb=test_rewrite_emb,
        eval_original_prompts=test_original_prompts,
        eval_rewrite_prompts=test_rewrite_prompts,
        pca_components=args.pca_components,
        tfidf_max_features=args.tfidf_max_features,
    )

    train_methods = (args.train_default, args.train_rewrite)
    test_methods = (args.test_default, args.test_rewrite)
    summary_rows = []
    candidate_rows = []
    for generator in args.generators:
        gen_val_train_features = dict(val_train_features)
        gen_val_features = dict(val_features)
        gen_test_train_features = dict(test_train_features)
        gen_test_features = dict(test_features)
        if args.include_target_t2i_embeddings:
            add_target_features_if_available(
                root,
                args.target_t2i_embedding_dir,
                generator,
                args.train_rewrite,
                args.test_rewrite,
                train_ids,
                test_ids,
                fit_idx,
                val_idx,
                gen_val_train_features,
                gen_val_features,
                args.target_pca_components,
            )
            add_target_features_if_available(
                root,
                args.target_t2i_embedding_dir,
                generator,
                args.train_rewrite,
                args.test_rewrite,
                train_ids,
                test_ids,
                fit_idx,
                None,
                gen_test_train_features,
                gen_test_features,
                args.target_pca_components,
            )

        y_fit = np.array(
            [train_scores[generator][args.train_rewrite][sample_id] - train_scores[generator][args.train_default][sample_id] for sample_id in fit_ids],
            dtype=float,
        )
        y_val = np.array(
            [train_scores[generator][args.train_rewrite][sample_id] - train_scores[generator][args.train_default][sample_id] for sample_id in val_ids],
            dtype=float,
        )
        val_default, val_rewrite = score_vectors(val_ids, generator, args.train_default, args.train_rewrite, train_scores)
        test_default_mean = fixed_mean(test_ids, generator, args.test_default, test_scores)
        test_rewrite_mean = fixed_mean(test_ids, generator, args.test_rewrite, test_scores)
        best = None
        for feature_name, X_fit_val in gen_val_train_features.items():
            X_val = gen_val_features[feature_name]
            for reg_name in args.regressors:
                try:
                    reg = make_regressor(reg_name)
                    reg.fit(X_fit_val, y_fit)
                    pred_val = np.asarray(reg.predict(X_val), dtype=float)
                except Exception as exc:
                    print(f"[skip] {generator} {feature_name} {reg_name}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                threshold, selected, val_routed = threshold_sweep_arrays(val_default, val_rewrite, pred_val)
                val_default_mean = float(val_default.mean())
                val_rewrite_mean = float(val_rewrite.mean())
                row = {
                    "generator": generator,
                    "feature_set": feature_name,
                    "regressor": reg_name,
                    "val_threshold": threshold,
                    "val_selected": selected,
                    "val_routed_qwen": val_routed,
                    "val_default_qwen": val_default_mean,
                    "val_rewrite_qwen": val_rewrite_mean,
                    "val_routed_minus_default": val_routed - val_default_mean,
                    "val_spearman": safe_corr(spearmanr, y_val, pred_val),
                }
                candidate_rows.append(row)
                key = (row["val_routed_qwen"], row["val_spearman"])
                if best is None or key > best[0]:
                    best = (key, row)
        if best is None:
            raise RuntimeError(f"No valid selector for {generator}")
        selected_cfg = best[1]
        feature_name = selected_cfg["feature_set"]
        reg_name = selected_cfg["regressor"]
        threshold = float(selected_cfg["val_threshold"])
        reg = make_regressor(reg_name)
        reg.fit(gen_test_train_features[feature_name], y_fit)
        pred_test = np.asarray(reg.predict(gen_test_features[feature_name]), dtype=float)
        test_routed, test_selected = route_with_threshold(test_ids, pred_test, threshold, generator, test_methods, test_scores)
        fixed_routed, fixed_selected = route_with_threshold(test_ids, pred_test, 0.0, generator, test_methods, test_scores)
        default = fixed_mean(test_ids, generator, args.test_default, test_scores)
        rewrite = fixed_mean(test_ids, generator, args.test_rewrite, test_scores)
        oracle = oracle_mean(test_ids, generator, test_methods, test_scores)
        summary_rows.append(
            {
                **selected_cfg,
                "test_routed_qwen": test_routed,
                "test_selected": test_selected,
                "test_fixed0_routed_qwen": fixed_routed,
                "test_fixed0_selected": fixed_selected,
                "test_default_qwen": default,
                "test_rewrite_qwen": rewrite,
                "test_oracle_qwen": oracle,
                "test_routed_minus_default": test_routed - default,
                "test_fixed0_minus_default": fixed_routed - default,
                "fit_count": len(fit_ids),
                "val_count": len(val_ids),
                "test_count": len(test_ids),
                "seed": args.seed,
            }
        )

        import joblib

        checkpoint_dir = out_dir / "checkpoints" / generator
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "regressor": reg,
                "feature_set": feature_name,
                "threshold": threshold,
                "generator": generator,
                "seed": args.seed,
                "fit_ids": fit_ids,
                "validation_ids": val_ids,
                "test_ids": test_ids,
                "test_predicted_gain": pred_test,
                "test_choose_rewrite": pred_test >= threshold,
            },
            checkpoint_dir / "selector.joblib",
        )

    suffix = "target_t2i" if args.include_target_t2i_embeddings else "qwen_surface_tfidf"
    gen_suffix = "all" if tuple(args.generators) == tuple(GENERATORS) else "_".join(args.generators).replace("-", "_")
    tag = f"_{safe_name(args.output_tag)}" if args.output_tag else ""
    prefix = f"{model_safe}_router_{suffix}{tag}_{gen_suffix}"
    summary = pd.DataFrame(summary_rows).sort_values("generator")
    candidates = pd.DataFrame(candidate_rows)
    summary.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    candidates.to_csv(out_dir / f"{prefix}_validation_candidates.csv", index=False)
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / f"{prefix}_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Strict Validation-Selected Gain Selector Tested on APBench\n\n")
        f.write("Selector config and threshold are selected only on a held-out non-APBench validation split.\n\n")
        f.write("| generator | feature_set | regressor | val routed | val-default | threshold | test routed | default | rewrite | oracle | routed-default | selected |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in summary.iterrows():
            f.write(
                f"| {r.generator} | {r.feature_set} | {r.regressor} | {r.val_routed_qwen:.6f} | {r.val_routed_minus_default:+.6f} | "
                f"{r.val_threshold:.6f} | {r.test_routed_qwen:.6f} | {r.test_default_qwen:.6f} | {r.test_rewrite_qwen:.6f} | "
                f"{r.test_oracle_qwen:.6f} | {r.test_routed_minus_default:+.6f} | {int(r.test_selected)} |\n"
            )
        if len(summary) == len(args.generators):
            f.write("\n## Mean Across Generators\n\n")
            for key in [
                "test_routed_qwen",
                "test_fixed0_routed_qwen",
                "test_default_qwen",
                "test_rewrite_qwen",
                "test_oracle_qwen",
                "test_routed_minus_default",
                "test_fixed0_minus_default",
            ]:
                f.write(f"- {key}: {summary[key].mean():.6f}\n")
    print(summary.to_string(index=False))
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
