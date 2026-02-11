import os
import json
from typing import List
import numpy as np

from metrics.utils.load_benchmark import load_benchmark_data

def split_benchmark_data(benchmark_data_path: str) -> dict:
    benchmark_data = load_benchmark_data(benchmark_data_path)
    easy_data = {'novice': [], 'expert': []}
    medium_data = {'novice': [], 'expert': []}
    hard_data = {'novice': [], 'expert': []}
    for item in benchmark_data:
        if item['class'] == 'intent_confirmed_all':
            easy_data[item['user_type']].append(item['id'])
        elif item['class'] == 'intent_confirmed_prompt_only':
            medium_data[item['user_type']].append(item['id'])
        elif item['class'] == 'intent_not_confirmed':
            hard_data[item['user_type']].append(item['id'])
    return {
        'easy': easy_data,
        'medium': medium_data,
        'hard': hard_data
    }
        

def merge_results(result_jsonl_list: List[str], merged_result_jsonl: str):
    merged_results = {}
    for result_jsonl in result_jsonl_list:
        if not os.path.exists(result_jsonl):
            continue
        with open(result_jsonl, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item['id'] not in merged_results:
                    merged_results[item['id']] = item
                else:
                    merged_results[item['id']].update(item)
    with open(merged_result_jsonl, 'w') as f:
        for item in merged_results.values():
            f.write(json.dumps(item) + '\n')

def print_eval_result(result_jsonl: str, need_print: bool = True):
    clip_class_scores = []
    clip_sentence_scores = []
    # internvl_class_scores = []
    # internvl_sentence_scores = []
    # qwen_class_scores = []
    qwen_sentence_scores = []
    result_data = {}
    with open(result_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('subject_clear') == True:
                clip_class_scores.extend(item.get('clip_class_scores', []))
            clip_sentence_scores.extend(item.get('clip_sentence_scores', []))
            qwen_sentence_scores.extend(item.get('qwen_sentence_scores', []))
    if clip_class_scores:
        result_data['clip_class_avg'] = sum(clip_class_scores) / len(clip_class_scores)
        result_data['clip_class_std'] = np.std(clip_class_scores)
        if need_print:
            print(f"CLIP Class-level Average Score: {sum(clip_class_scores) / len(clip_class_scores):.4f}")
            print(f"CLIP Class-level Standard Deviation: {np.std(clip_class_scores):.4f}")
    if qwen_sentence_scores:
        result_data['qwen_sentence_avg'] = sum(qwen_sentence_scores) / len(qwen_sentence_scores)
        if need_print:
            print(f"Qwen Sentence-level Average Score: {sum(qwen_sentence_scores) / len(qwen_sentence_scores):.4f}")
    return result_data
def print_eval_result_by_difficulty_and_user_type(result_jsonl: str, benchmark_data_path: str, need_print: bool = True):
    split_data = split_benchmark_data(benchmark_data_path)
    difficulty_levels = ['easy', 'medium', 'hard']
    result_data = {}
    for level in difficulty_levels:
        for user_type in ['novice', 'expert']:
            ids = set(split_data[level][user_type])
            clip_class_scores = []
            qwen_sentence_scores = []
            with open(result_jsonl, 'r') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item['id'] in ids and item.get('subject_clear') == True:
                        clip_class_scores.extend(item.get('clip_class_scores', []))
                    if item['id'] in ids:
                        qwen_sentence_scores.extend(item.get('qwen_sentence_scores', []))
            if clip_class_scores:
                result_data[(level, user_type)] = {
                    'clip_class_avg': sum(clip_class_scores) / len(clip_class_scores),
                    'clip_class_std': np.std(clip_class_scores)
                }
                if need_print:
                    print(f"[{level.capitalize()} - {user_type.capitalize()}] CLIP Class-level Average Score: {sum(clip_class_scores) / len(clip_class_scores):.4f}")
                    print(f"[{level.capitalize()} - {user_type.capitalize()}] CLIP Class-level Standard Deviation: {np.std(clip_class_scores):.4f}")
            
            if qwen_sentence_scores:
                # 确保键存在，防止之前只存在 clip_class 字段时发生 KeyError
                if (level, user_type) not in result_data:
                    result_data[(level, user_type)] = {}
                result_data[(level, user_type)]['qwen_sentence_avg'] = sum(qwen_sentence_scores) / len(qwen_sentence_scores)
                if need_print:
                    print(f"[{level.capitalize()} - {user_type.capitalize()}] Qwen Sentence-level Average Score: {sum(qwen_sentence_scores) / len(qwen_sentence_scores):.4f}")
            if need_print:
                print('---'*20)

    return result_data
    

def process_single_result_dir(result_dir: str, need_print: bool = True) -> List[str]:
    merged_path = os.path.join(result_dir, "merged_result.jsonl")
    # gather jsonl files except merged_result.jsonl
    jsonl_files = []
    if os.path.isdir(result_dir):
        for fname in os.listdir(result_dir):
            if not fname.endswith('.jsonl'):
                continue
            if fname == os.path.basename(merged_path):
                continue
            jsonl_files.append(os.path.join(result_dir, fname))
    if not jsonl_files:
        print(f"No jsonl result files found in {result_dir}")
        return []
    merge_results(result_jsonl_list=jsonl_files, merged_result_jsonl=merged_path)
    if need_print:
        print(f"\nDirectory: {result_dir}")
        print(f"Evaluation results saved to {merged_path}")
        print('=='*20 + ' Evaluation Summary ' + '='*20)
        print(f"Generated images directory: {result_dir}")
        print_eval_result(merged_path)
        print('=='*20 + ' Evaluation Summary by Difficulty and User Type ' + '='*20)
        print_eval_result_by_difficulty_and_user_type(merged_path, 'data/benchmark/auth_prompt_V1_0_0.json', need_print=need_print)
    else:
        results = print_eval_result(merged_path, need_print=need_print)
        results['by_difficulty_and_user_type'] = print_eval_result_by_difficulty_and_user_type(merged_path, 'data/benchmark/auth_prompt_V1_0_0.json', need_print=need_print)
    # 返回合并后的每一条记录，供批量模式按列输出
    merged_lines = []
    with open(merged_path, 'r') as f:
        for line in f:
            merged_lines.append(line.strip())
    assert len(merged_lines) == 2048, f"Expected 2048 results, got {len(merged_lines)} in {result_dir}"
    if not need_print:
        return results

def main(result_dir: str, batch: bool = False):
    if batch:
        if not os.path.isdir(result_dir):
            print(f"{result_dir} is not a directory")
            return
        all_results = {}
        for entry in sorted(os.listdir(result_dir)):
            sub = os.path.join(result_dir, entry)
            if os.path.isdir(sub):
                res = process_single_result_dir(sub, need_print=False)
                all_results[entry] = res or {}
        if not all_results:
            print("No subdirectories with results found.")
            return

        cols = ["sd21", "comat21", "sd3", "flux1", "pixart-sigma", "infi", "showo2"]

        def fmt(v):
            if v is None:
                return "N/A"
            if isinstance(v, str):
                return v
            try:
                return f"{float(v):.4f}"
            except Exception:
                return str(v)

        # Summary transposed: rows = metrics, cols = datasets
        metrics = [("CLIP_Class_Avg", "clip_class_avg"), ("CLIP_Class_Std", "clip_class_std"), ("Qwen_Sentence_Avg", "qwen_sentence_avg")]
        header = ["Metric"] + cols
        table = []
        for mname, key in metrics:
            row = [mname]
            for c in cols:
                res = all_results.get(c, {})
                val = res.get(key, None)
                row.append(fmt(val))
            table.append(row)
        col_widths = [max(len(header[i]), *(len(r[i]) for r in table)) for i in range(len(header))]
        print(" | ".join(header[i].ljust(col_widths[i]) for i in range(len(header))))
        print("-+-".join("-" * col_widths[i] for i in range(len(header))))
        for r in table:
            print(" | ".join(r[i].ljust(col_widths[i]) for i in range(len(r))))

        # Per difficulty/user-type transposed
        user_types = ['novice', 'expert']
        difficulty_levels = ['easy', 'medium', 'hard']
        for level in difficulty_levels:
            for utype in user_types:
                title = f"{level.capitalize()} - {utype.capitalize()}"
                print("\n" + title)
                sub_table = []
                for mname, key in metrics:
                    row = [mname]
                    for c in cols:
                        res = all_results.get(c, {})
                        by = res.get('by_difficulty_and_user_type', {})
                        sub = by.get((level, utype), {})
                        row.append(fmt(sub.get(key) if sub else None))
                    sub_table.append(row)
                sub_header = ["Metric"] + cols
                sub_col_widths = [max(len(sub_header[i]), *(len(r[i]) for r in sub_table)) for i in range(len(sub_header))]
                print(" | ".join(sub_header[i].ljust(sub_col_widths[i]) for i in range(len(sub_header))))
                print("-+-".join("-" * sub_col_widths[i] for i in range(len(sub_header))))
                for r in sub_table:
                    print(" | ".join(r[i].ljust(sub_col_widths[i]) for i in range(len(r))))
    else:
        process_single_result_dir(result_dir)

if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if not args:
        print("Usage: python print_res.py <result_dir>     # 单目录处理")
        print("   or: python print_res.py -b <result_root>  # 对 result_root 下每个子目录批量处理并逐行输出")
        sys.exit(1)
    if args[0] == '-b':
        if len(args) < 2:
            print("请提供要批量处理的结果根目录")
            sys.exit(1)
        result_root = args[1]
        main(result_root, batch=True)
    else:
        result_dir = args[0]
        main(result_dir)