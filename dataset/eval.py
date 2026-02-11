import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    CLIPProcessor, CLIPModel,
    Qwen3VLForConditionalGeneration, AutoProcessor
)

import json
import time
from tqdm import tqdm
from PIL import Image
from glob import glob
from omegaconf import OmegaConf
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import os
from collections import defaultdict
import numpy as np
from typing import List
import re
import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from models.utils.internvl_load_image import load_image
from metrics.utils.load_benchmark import load_benchmark_data

from huggingface_hub import login
from tqdm import tqdm
import os

login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

class CLIPEval:
    def __init__(self, device="cuda"):
        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.device = device

    @torch.no_grad()
    def class_level_eval(self, images: List[Image.Image], intent: str, candidate_intents: List[str]) -> List[float]:
        assert intent in candidate_intents, "The ground truth intent must be in the candidate intents."
        index = candidate_intents.index(intent)
        candidate_intents = [f'This is a photo of {label}.' for label in candidate_intents]
        inputs = self.processor(
            text=candidate_intents,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        scores = probs[:, index].cpu().tolist()
        return scores
    
    @torch.no_grad()
    def sentence_level_eval(self, images: List[Image.Image], intent: str):
        candidate_intents = [intent]
        inputs = self.processor(
            text=candidate_intents,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        return probs.flatten().cpu().tolist()
    
class InternVLEval:
    def __init__(
            self, 
            model_name="OpenGVLab/InternVL3_5-4B-Flash",
            device="cuda",
            dtype=torch.bfloat16
            ):
        self.internvl_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.internvl_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval().to(device)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def eval(self, images: List[Image.Image], intent: str, class_level: bool=False):
        if class_level:
            intent = f"This is a photo of {intent}."
        scores = []
        for img in images:
            pixel_values = load_image(img).to(self.device, dtype=self.dtype)
            generation_config = dict(max_new_tokens=1024, do_sample=True)

            question = f"<image>\nOnly give a score from 0 to 1 indicating how well the image matches the intent: {intent}"
            response = self.internvl_model.chat(
                self.internvl_tokenizer,
                pixel_values,
                question,
                generation_config
            )
            try:
                # 提取第一个出现的0到1之间的浮点数
                match = re.search(r"([01](?:\.\d+)?)", response)
                if match:
                    score = float(match.group(1))
                else:
                    score = 0.0
                    logger.warning(f"No valid score found in response for intent '{intent}': {response}")
            except Exception:
                score = 0.0
                logger.warning(f"Error parsing score from response for intent '{intent}': {response}")
            scores.append(score)
        return scores
    
class QwenEval:
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.device = device

    @torch.no_grad()
    def eval(self, images: List[Image.Image], intent: str, class_level: bool=False):
        if class_level:
            intent = f"This is a photo of {intent}."
        scores = []
        for img in images:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": f"Only give a score from 0 to 1 indicating how well the image matches the intent: {intent}."},
                    {"type": "image", "image": img}
                ]}
            ]
            inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = output_text[0]
            try:
                # 提取第一个出现的0到1之间的浮点数
                match = re.search(r"([01](?:\.\d+)?)", response)
                if match:
                    score = float(match.group(1))
                else:
                    score = 0.0
                    logger.warning(f"No valid score found in response for intent '{intent}': {response}")
            except Exception:
                score = 0.0
                logger.warning(f"Error parsing score from response for intent '{intent}': {response}")
            scores.append(score)
        return scores  

def eval(
        images_dir: str,
        result_jsonl: str,
        evaluation_model: str = 'clip',
        benchmark_path: str = 'data/benchmark/auth_prompt_V1_0_0.json',
):
    candidate_intents  = OmegaConf.load("configs/classname/imagenet.yaml")['classes']
    if not os.path.exists(os.path.dirname(result_jsonl)):
        os.makedirs(os.path.dirname(result_jsonl))
    benchmark_data = load_benchmark_data(benchmark_path)
    image_dirs = glob(os.path.join(images_dir, "*"))
    image_dirs.sort(key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]))
    assert len(image_dirs) == len(benchmark_data), f"The number of images({len(image_dirs)}) must match the number of benchmark data entries({len(benchmark_data)})."
    if os.path.exists(result_jsonl):
        with open(result_jsonl, 'r') as f:
            existing_ids = {json.loads(line.strip())['id'] for line in f}
            existing_indexes = {i for i, item in enumerate(benchmark_data) if item['id'] in existing_ids}
        benchmark_data = [item for item in benchmark_data if item['id'] not in existing_ids]
        image_dirs = [d for i, d in enumerate(image_dirs) if i not in existing_indexes]
        print(f"Resuming evaluation. {len(existing_ids)} entries already exist. {len(benchmark_data)} entries remaining.")
    if evaluation_model == 'clip':
        clip_eval_model = CLIPEval(device="cuda")
        for img_dir, item in tqdm(zip(image_dirs, benchmark_data), total=len(benchmark_data), desc="Evaluating", ncols=80):
            img_paths = []
            for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
                img_paths.extend(glob(os.path.join(img_dir, f"*.{ext}")))
            img_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            try:
                images = [Image.open(path) for path in img_paths]
            except Exception as e:
                print(f"Error loading images from {img_dir}: {e}")
                continue
            class_intent = item['intent']
            sentence_intent = item['sentence_intent']
            if item['subject_clear'] == True:
                clip_class_scores = clip_eval_model.class_level_eval(images, class_intent, candidate_intents)
                clip_sentence_scores = clip_eval_model.sentence_level_eval(images, sentence_intent)
                result = {
                    'id': item['id'],
                    'user_type': item['user_type'],
                    'subject_clear': item['subject_clear'],
                    'clip_class_scores': clip_class_scores,
                    'clip_sentence_scores': clip_sentence_scores,
                }
                with open(result_jsonl, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                clip_sentence_scores = clip_eval_model.sentence_level_eval(images, sentence_intent)
                result = {
                    'id': item['id'],
                    'user_type': item['user_type'],
                    'subject_clear': item['subject_clear'],
                    'clip_sentence_scores': clip_sentence_scores,
                }
                with open(result_jsonl, 'a') as f:
                    f.write(json.dumps(result) + '\n')
        del clip_eval_model
    elif evaluation_model == 'internvl':    
        internvl_eval_model = InternVLEval()
        for img_dir, item in tqdm(zip(image_dirs, benchmark_data), total=len(benchmark_data), desc="Evaluating", ncols=80):
            img_paths = []
            for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
                img_paths.extend(glob(os.path.join(img_dir, f"*.{ext}")))
            img_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            try:
                images = [Image.open(path) for path in img_paths]
            except Exception as e:
                print(f"Error loading images from {img_dir}: {e}")
                continue
            class_intent = item['intent']
            sentence_intent = item['sentence_intent']
            if item['subject_clear'] == True:
                internvl_class_scores = internvl_eval_model.eval(images, class_intent, class_level=True)
                internvl_sentence_scores = internvl_eval_model.eval(images, sentence_intent)
                result = {
                    'id': item['id'],
                    'user_type': item['user_type'],
                    'subject_clear': item['subject_clear'],
                    'internvl_class_scores': internvl_class_scores,
                    'internvl_sentence_scores': internvl_sentence_scores,
                }
                with open(result_jsonl, 'a') as f:
                    f.write(json.dumps(result) + '\n')
            else:
                internvl_sentence_scores = internvl_eval_model.eval(images, sentence_intent)
                result = {
                    'id': item['id'],
                    'user_type': item['user_type'],
                    'subject_clear': item['subject_clear'],
                    'clip_sentence_scores': clip_sentence_scores,
                    'internvl_sentence_scores': internvl_sentence_scores,
                    'qwen_sentence_scores': qwen_sentence_scores,
                }
                with open(result_jsonl, 'a') as f:
                    f.write(json.dumps(result) + '\n')   
        del internvl_eval_model     
    elif evaluation_model == 'qwen':
        qwen_eval_model = QwenEval()    
        for img_dir, item in tqdm(zip(image_dirs, benchmark_data), total=len(benchmark_data), desc="Evaluating", ncols=80):
            img_paths = []
            for ext in ("png", "jpg", "jpeg", "PNG", "JPG", "JPEG"):
                img_paths.extend(glob(os.path.join(img_dir, f"*.{ext}")))
            img_paths.sort(key=lambda x: int(os.path.basename(x).split(".")[0]))
            try:
                images = [Image.open(path) for path in img_paths]
            except Exception as e:
                print(f"Error loading images from {img_dir}: {e}")
                continue
            sentence_intent = item['sentence_intent']
            qwen_sentence_scores = qwen_eval_model.eval(images, sentence_intent)
            result = {
                'id': item['id'],
                'user_type': item['user_type'],
                'subject_clear': item['subject_clear'],
                'qwen_sentence_scores': qwen_sentence_scores,
            }
            with open(result_jsonl, 'a') as f:
                f.write(json.dumps(result) + '\n')
        del qwen_eval_model

def merge_results(result_jsonl_list: List[str], merged_result_jsonl: str):
    merged_results = {}
    for result_jsonl in result_jsonl_list:
        with open(result_jsonl, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                if item['id'] not in merged_results:
                    merged_results[item['id']] = item
                else:
                    merged_results[item['id']].update(item)
    with open(merged_result_jsonl, 'w') as f:
        for item in merged_results.values():
            f.write(json.dumps(item) + '\n')

def main(images_dir: str, result_dir: str, benchmark_path: str):
    eval(
        images_dir=images_dir,
        result_jsonl=os.path.join(result_dir, "clip.jsonl"),
        benchmark_path=benchmark_path,
        evaluation_model='clip'
    )
    eval(
        images_dir=images_dir,
        result_jsonl=os.path.join(result_dir, "qwen.jsonl"),
        benchmark_path=benchmark_path,
        evaluation_model='qwen'
    )
    # result_jsonl = os.path.join(result_dir, "merged_result.jsonl")
    # merge_results(
    #     result_jsonl_list=[
    #         os.path.join(result_dir, "clip.jsonl"),
    #         os.path.join(result_dir, "qwen.jsonl"),
    #     ],
    #     merged_result_jsonl=result_jsonl
    # )

    # print(f"Evaluation results saved to {result_jsonl}")
    # print('=='*20 + ' Evaluation Summary ' + '='*20)
    # print(f"Generated images directory: {images_dir}")
    # print_eval_result(result_jsonl)

def print_eval_result(result_jsonl: str):
    clip_class_scores = []
    clip_sentence_scores = []
    # internvl_class_scores = []
    # internvl_sentence_scores = []
    # qwen_class_scores = []
    qwen_sentence_scores = []
    with open(result_jsonl, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            if item['subject_clear'] == True:
                clip_class_scores.extend(item['clip_class_scores'])
                # internvl_class_scores.extend(item['internvl_class_scores'])
                # qwen_class_scores.extend(item['qwen_class_scores'])
            clip_sentence_scores.extend(item['clip_sentence_scores'])
            # internvl_sentence_scores.extend(item['internvl_sentence_scores'])
            qwen_sentence_scores.extend(item['qwen_sentence_scores'])
    print(f"CLIP Class-level Average Score: {sum(clip_class_scores) / len(clip_class_scores):.4f}")
    print(f"CLIP Class-level Standard Deviation: {np.std(clip_class_scores):.4f}")
    # print(f"InternVL Class-level Average Score: {sum(internvl_class_scores) / len(internvl_class_scores):.4f}")
    # print(f"Qwen Class-level Average Score: {sum(qwen_class_scores) / len(qwen_class_scores):.4f}")
    print(f"CLIP Sentence-level Average Score: {sum(clip_sentence_scores) / len(clip_sentence_scores):.4f}")
    # print(f"InternVL Sentence-level Average Score: {sum(internvl_sentence_scores) / len(internvl_sentence_scores):.4f}")
    print(f"Qwen Sentence-level Average Score: {sum(qwen_sentence_scores) / len(qwen_sentence_scores):.4f}")

    
def test():
    from omegaconf import OmegaConf
    from PIL import Image
    # candidate_intents  = OmegaConf.load("configs/classname/openimages.yaml")['classes']
    # clip_eval_model = CLIPEval(device="cuda")
    # internvl_eval_model = InternVLEval()
    qwen_eval_model = QwenEval()
    # Example usage
    image_dir = "data/benchmark/generated_images/comat21/prompt_10"
    class_intent = "tiger"
    sentence_intent = "an image of a tiger's eyes as the central element of a landing page design"
    from glob import glob
    image_paths = glob(os.path.join(image_dir, "*.png"))
    images = [Image.open(path) for path in image_paths]
    # scores = clip_eval_model.class_level_eval(images, intent, candidate_intents)
    # print(f"Class-level evaluation scores: {scores}")
    # scores = clip_eval_model.sentence_level_eval(images, sentence_intent)
    # print(f"Sentence-level evaluation scores: {scores}")
    # scores = internvl_eval_model.eval(images, sentence_intent)
    scores = qwen_eval_model.eval(images, sentence_intent)
    print(f"InternVL evaluation scores: {scores}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate generated images against user intents.")
    parser.add_argument("--images_dir", type=str, help="Directory containing generated images.", required=True)
    parser.add_argument("--result_dir", type=str, help="Directory to save the evaluation results in jsonl format.", required=True)
    parser.add_argument("--benchmark_path", type=str, help="Path to the benchmark data json file.", default='data/benchmark/auth_prompt_V1_0_0.json')
    
    args = parser.parse_args()
    
    main(
        images_dir=args.images_dir,
        result_dir=args.result_dir,
        benchmark_path=args.benchmark_path
    )
    # test()

