import os
import torch
import lightning as L
from typing import List, Sequence
from datasets import Dataset
from datasets import load_dataset
from tqdm import tqdm
import csv
import time
from omegaconf import DictConfig
import hydra

from metrics.utils.load_benchmark import (
    PROJECT_ROOT,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'for_cb')

def load_texts(file_path: str | Sequence[str]) -> List[str]:
    if isinstance(file_path, str):
        file_path = [file_path]
    texts = []
    for path in file_path:
        with open(path, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])
    return texts

def refine_prompt_for_cb(prompts: List[str], output_dir: str) -> List[str]:
    if os.path.exists(os.path.join(output_dir, f"llama3_grpo.txt")):
        print("Refined prompts already exist, loading from file.")
        refined_data_path = os.path.join(output_dir, f"llama3_grpo.txt")
        with open(refined_data_path, 'r', encoding='utf-8') as f:
            refined_prompts = [line.strip() for line in f if line.strip()]
        assert len(prompts) == len(refined_prompts), "原始提示词和精炼后提示词数量不匹配"
        return refined_prompts
    from models.components.refiner.llmrefiner import AutoModelLLMRefiner
    ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/llama3_grpo/version_0/refiner-epoch=00-train_loss=-0.00.ckpt")
    refiner = AutoModelLLMRefiner(model_name="meta-llama/Llama-3.1-8B-Instruct")
    refiner.load_checkpoint(ckpt_path)
    def save_function(refined_texts: List[str]):
        assert len(prompts) == len(refined_texts)
        refined_data_path = os.path.join(output_dir, f"llama3_grpo.txt")
        with open(refined_data_path, 'w', encoding='utf-8') as f:
            for text in refined_texts:
                f.write(text + '\n')
    refiner.generate_and_save(prompts, save_function)

def gen(
        model_name: str,
        data_path: str | Sequence[str],
        img_size: int,
        num_samples: int,
        output_dirs: Sequence[str],
        start: int,
        end: int,
        refined: bool = False,
        **kwargs):
    """
    根据文本生成图片，每个文本生成 num_samples 张图片，分别保存在 output_dirs，
    文件名格式为 prompt+六位数字，比如 a green bench and a blue bowl_000000.png
    """
    if isinstance(data_path, str):
        data_path = [data_path]
    if isinstance(output_dirs, str):
        output_dirs = [output_dirs]
    assert len(data_path) == len(output_dirs), "data_path 和 output_dirs 长度必须一致"

    match model_name:
        case "sd21":
            from diffusers import StableDiffusionPipeline
            import torch

            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-1",
                    torch_dtype=torch.bfloat16,
                )
            except Exception as e:
                pipe = StableDiffusionPipeline.from_pretrained(
                    "hanker917/stable-diffusion-2-1",
                    torch_dtype=torch.bfloat16,
                )
            pipe = pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)


            for path, out_dir in zip(data_path, output_dirs):
                os.makedirs(out_dir, exist_ok=True)
                texts = load_texts(path)
                if refined:
                    texts = refine_prompt_for_cb(texts)
                texts = texts[start:end]
                print(f"Loaded {len(texts)} texts from {path}, generating images to {out_dir}")
                for text in tqdm(texts, desc=f"Generating images for {path}"):
                    images = pipe(
                        text, 
                        num_images_per_prompt=num_samples,
                        output_type="pil",
                        height=img_size, 
                        width=img_size,
                        **kwargs).images
                    idx = 0
                    for image in images:
                        safe_prompt = text.replace('/', '_').replace('\\', '_').replace(':', '_')
                        filename = f"{safe_prompt}_{idx:06d}.png"
                        save_path = os.path.join(out_dir, filename)
                        image.save(save_path)
                        idx += 1
                print(f"生成的图片保存在 {out_dir}")
        case 'comat21':
            from models.comat import CoMat
            import torch
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, 'configs/comat/model/sd21.yaml'))
            model_cfg.do_classifier_free_guidance = True
            
            try:
                model = CoMat(**model_cfg).to(dtype=torch.bfloat16).to("cuda")
            except Exception as e:
                model_cfg.pretrain_model = "hanker917/stable-diffusion-2-1"
                model = CoMat(**model_cfg).to(dtype=torch.bfloat16).to("cuda")
            model.load_checkpoint(os.path.join(PROJECT_ROOT, "checkpoints/CoMat/comat21/step=2000-g_loss=6.6597-d_loss=0.3280.ckpt"))            
            model.pipeline.set_progress_bar_config(disable=True)
            for path, out_dir in zip(data_path, output_dirs):
                os.makedirs(out_dir, exist_ok=True)
                texts = load_texts(path)
                if refined:
                    texts = refine_prompt_for_cb(texts)
                texts = texts[start:end]
                print(f"Loaded {len(texts)} texts from {path}, generating images to {out_dir}")
                for text in tqdm(texts, desc=f"Generating images for {path}"):
                    images = model.pipeline(
                        text, 
                        num_images_per_prompt=num_samples,
                        output_type="pil",
                        height=img_size, 
                        width=img_size,
                        **kwargs).images
                    idx = 0
                    for image in images:
                        safe_prompt = text.replace('/', '_').replace('\\', '_').replace(':', '_')
                        filename = f"{safe_prompt}_{idx:06d}.png"
                        save_path = os.path.join(out_dir, filename)
                        image.save(save_path)
                        idx += 1
                print(f"生成的图片保存在 {out_dir}")
        case _:
            raise ValueError(f"不支持的模型名称: {model_name}")
        
if __name__ == "__main__":
    L.seed_everything(995)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='模型名称，比如 sd21, comat21')
    parser.add_argument('--data_path', type=str, nargs='+', required=True, help='文本文件路径，可以是多个')
    parser.add_argument('--img_size', type=int, default=512, help='生成图片的尺寸，默认为 512')
    parser.add_argument('--num_samples', type=int, default=1, help='每个文本生成的图片数量，默认为 1')
    parser.add_argument('--output_dirs', type=str, nargs='+', required=True, help='输出目录，可以是多个，数量应与 data_path 一致')
    parser.add_argument('--start', type=int, default=0, help='从第几个文本开始生成，默认为 0')
    parser.add_argument('--end', type=int, default=None, help='到第几个文本结束，默认为 None，表示生成到最后')
    parser.add_argument('--refined', type=bool, default=False, help='是否先对提示词进行精炼，默认为 False')
    args = parser.parse_args()
    gen(
        model_name=args.model_name,
        data_path=args.data_path,
        img_size=args.img_size,
        num_samples=args.num_samples,
        output_dirs=args.output_dirs,
        start=args.start,
        end=args.end,
        refined=args.refined,
    )