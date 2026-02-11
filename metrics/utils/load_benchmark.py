import json
import os

DATA_FILE_PATH = 'data/benchmark/auth_prompt_V1_0_0.json'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

REFINED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/benchmark/refined_prompts")

def load_benchmark_data(file_path=DATA_FILE_PATH):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_refined_data_dir(refiner_type: str) -> str:
    refined_data_path = os.path.join(REFINED_DATA_DIR, f"{refiner_type}.json")
    return refined_data_path

def load_refined_data(refiner_type: str) -> list:
    refined_data_path = get_refined_data_dir(refiner_type)
    with open(refined_data_path, 'r', encoding='utf-8') as f:
        refined_data = json.load(f)
    for item in refined_data:
        assert item['refiner_type'] == refiner_type
        item['prompt'] = item['refined_prompt']
    return refined_data
    