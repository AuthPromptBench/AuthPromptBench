import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics.utils.load_benchmark import DATA_FILE_PATH, load_benchmark_data


DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
USER_TYPES = ["novice", "expert"]
DEFAULT_COLUMNS = ["sd21", "comat21", "sd3", "flux1", "pixart-sigma", "infi", "showo2"]
SUMMARY_METRICS = [
    ("CLIP_Class_Avg", "clip_class_avg"),
    ("CLIP_Class_Std", "clip_class_std"),
    ("Qwen_Sentence_Avg", "qwen_sentence_avg"),
]


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_benchmark_data(benchmark_data_path: str = DATA_FILE_PATH) -> Dict[str, Dict[str, List[str]]]:
    split_data = {level: {user_type: [] for user_type in USER_TYPES} for level in DIFFICULTY_LEVELS}
    for item in load_benchmark_data(benchmark_data_path):
        difficulty = item.get("challenge")
        if difficulty not in DIFFICULTY_LEVELS:
            raise ValueError(
                f"Benchmark record {item.get('id')} has invalid or missing challenge: {difficulty!r}"
            )
        user_type = item.get("user_type")
        if user_type not in USER_TYPES:
            raise ValueError(
                f"Benchmark record {item.get('id')} has invalid or missing user_type: {user_type!r}"
            )
        split_data[difficulty][user_type].append(item["id"])
    return split_data


def merge_results(result_jsonl_list: List[str], merged_result_jsonl: str) -> None:
    merged_results = OrderedDict()
    for result_jsonl in result_jsonl_list:
        if not os.path.exists(result_jsonl):
            continue
        for item in read_jsonl(result_jsonl):
            item_id = item["id"]
            if item_id not in merged_results:
                merged_results[item_id] = item
            else:
                merged_results[item_id].update(item)
    write_jsonl(merged_result_jsonl, merged_results.values())


def collect_scores(rows: Iterable[dict], ids: Optional[set] = None) -> Dict[str, List[float]]:
    clip_class_scores = []
    qwen_sentence_scores = []

    for item in rows:
        if ids is not None and item.get("id") not in ids:
            continue
        if item.get("subject_clear") is True:
            clip_class_scores.extend(item.get("clip_class_scores", []))
        qwen_sentence_scores.extend(item.get("qwen_sentence_scores", []))

    return {
        "clip_class_scores": clip_class_scores,
        "qwen_sentence_scores": qwen_sentence_scores,
    }


def summarize_scores(scores: Dict[str, List[float]]) -> Dict[str, float]:
    result = {}
    clip_class_scores = scores["clip_class_scores"]
    qwen_sentence_scores = scores["qwen_sentence_scores"]

    if clip_class_scores:
        result["clip_class_avg"] = sum(clip_class_scores) / len(clip_class_scores)
        result["clip_class_std"] = float(np.std(clip_class_scores))
    if qwen_sentence_scores:
        result["qwen_sentence_avg"] = sum(qwen_sentence_scores) / len(qwen_sentence_scores)
    return result


def print_summary(result: Dict[str, float], prefix: str = "") -> None:
    label = f"{prefix} " if prefix else ""
    if "clip_class_avg" in result:
        print(f"{label}CLIP Class-level Average Score: {result['clip_class_avg']:.4f}")
        print(f"{label}CLIP Class-level Standard Deviation: {result['clip_class_std']:.4f}")
    if "qwen_sentence_avg" in result:
        print(f"{label}Qwen Sentence-level Average Score: {result['qwen_sentence_avg']:.4f}")


def print_eval_result(result_jsonl: str, need_print: bool = True) -> Dict[str, float]:
    result = summarize_scores(collect_scores(read_jsonl(result_jsonl)))
    if need_print:
        print_summary(result)
    return result


def print_eval_result_by_difficulty_and_user_type(
    result_jsonl: str,
    benchmark_data_path: str = DATA_FILE_PATH,
    need_print: bool = True,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    split_data = split_benchmark_data(benchmark_data_path)
    rows = list(read_jsonl(result_jsonl))
    result_data = {}

    for level in DIFFICULTY_LEVELS:
        for user_type in USER_TYPES:
            ids = set(split_data[level][user_type])
            result = summarize_scores(collect_scores(rows, ids))
            if result:
                result_data[(level, user_type)] = result
            if need_print:
                print_summary(result, prefix=f"[{level.capitalize()} - {user_type.capitalize()}]")
                print("---" * 20)

    return result_data


def result_files(result_dir: str, merged_path: str) -> List[str]:
    if not os.path.isdir(result_dir):
        return []
    return [
        os.path.join(result_dir, fname)
        for fname in sorted(os.listdir(result_dir))
        if fname.endswith(".jsonl") and fname != os.path.basename(merged_path)
    ]


def process_single_result_dir(
    result_dir: str,
    need_print: bool = True,
    expected_count: Optional[int] = 2048,
) -> Dict[str, object]:
    merged_path = os.path.join(result_dir, "merged_result.jsonl")
    jsonl_files = result_files(result_dir, merged_path)
    if not jsonl_files:
        print(f"No jsonl result files found in {result_dir}")
        return {}

    merge_results(result_jsonl_list=jsonl_files, merged_result_jsonl=merged_path)
    if need_print:
        print(f"\nDirectory: {result_dir}")
        print(f"Evaluation results saved to {merged_path}")
        print("==" * 20 + " Evaluation Summary " + "==" * 20)
        print(f"Generated images directory: {result_dir}")

    results = print_eval_result(merged_path, need_print=need_print)
    if need_print:
        print("==" * 20 + " Evaluation Summary by Difficulty and User Type " + "==" * 20)
    results["by_difficulty_and_user_type"] = print_eval_result_by_difficulty_and_user_type(
        merged_path,
        DATA_FILE_PATH,
        need_print=need_print,
    )

    merged_count = sum(1 for _ in read_jsonl(merged_path))
    if expected_count is not None and merged_count != expected_count:
        raise AssertionError(f"Expected {expected_count} results, got {merged_count} in {result_dir}")
    return results


def fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def print_table(header: List[str], table: List[List[str]]) -> None:
    col_widths = [max(len(header[i]), *(len(row[i]) for row in table)) for i in range(len(header))]
    print(" | ".join(header[i].ljust(col_widths[i]) for i in range(len(header))))
    print("-+-".join("-" * col_widths[i] for i in range(len(header))))
    for row in table:
        print(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(row))))


def metric_table(all_results: Dict[str, Dict[str, object]], columns: List[str]) -> List[List[str]]:
    table = []
    for metric_name, key in SUMMARY_METRICS:
        row = [metric_name]
        for column in columns:
            row.append(fmt(all_results.get(column, {}).get(key)))
        table.append(row)
    return table


def difficulty_metric_table(
    all_results: Dict[str, Dict[str, object]],
    columns: List[str],
    level: str,
    user_type: str,
) -> List[List[str]]:
    table = []
    for metric_name, key in SUMMARY_METRICS:
        row = [metric_name]
        for column in columns:
            by_group = all_results.get(column, {}).get("by_difficulty_and_user_type", {})
            group_result = by_group.get((level, user_type), {})
            row.append(fmt(group_result.get(key) if group_result else None))
        table.append(row)
    return table


def process_batch(result_root: str, columns: List[str]) -> None:
    if not os.path.isdir(result_root):
        print(f"{result_root} is not a directory")
        return

    all_results = {}
    for entry in sorted(os.listdir(result_root)):
        subdir = os.path.join(result_root, entry)
        if os.path.isdir(subdir):
            all_results[entry] = process_single_result_dir(subdir, need_print=False) or {}
    if not all_results:
        print("No subdirectories with results found.")
        return

    print_table(["Metric"] + columns, metric_table(all_results, columns))
    for level in DIFFICULTY_LEVELS:
        for user_type in USER_TYPES:
            print(f"\n{level.capitalize()} - {user_type.capitalize()}")
            print_table(
                ["Metric"] + columns,
                difficulty_metric_table(all_results, columns, level, user_type),
            )


def main(result_dir: str, batch: bool = False, expected_count: Optional[int] = 2048) -> None:
    if batch:
        process_batch(result_dir, DEFAULT_COLUMNS)
    else:
        process_single_result_dir(result_dir, expected_count=expected_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge and summarize benchmark evaluation JSONL files.")
    parser.add_argument("result_dir", help="Result directory, or result root when using --batch.")
    parser.add_argument("-b", "--batch", action="store_true", help="Summarize each subdirectory under result_dir.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=2048,
        help="Expected merged result count. Use 0 to skip this check.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.result_dir, batch=args.batch, expected_count=args.expected_count or None)
