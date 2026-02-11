import os
import json
from tqdm import tqdm
import math

from metrics.utils.load_benchmark import (
    DATA_FILE_PATH,
    PROJECT_ROOT,
)

from huggingface_hub import login

login(token="")

def load_geneval_data(data_file_path: str) -> list:
    with open(data_file_path, 'r', encoding='utf-8') as f:
        datas = [json.loads(line) for line in f]
    return datas

def prepare_prompt_folder(output_dir: str, prompt_idx: int, metadata: dict):
    folder = os.path.join(output_dir, f"{prompt_idx:05d}")
    samples_dir = os.path.join(folder, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    meta_path = os.path.join(folder, "metadata.jsonl")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=4)
            mf.write("\n")
    return folder, samples_dir

def generate_images(
        model_name: str,
        metadatas: list,
        output_dir: str,
        images_per_prompt: int = 4,
        user_refiner: bool = False,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    key = 'prompt' if not user_refiner else 'refined_prompt'
    print(f"Using key '{key}' for prompts.")

    match model_name:
        case 'sd3':
            from diffusers import StableDiffusion3Pipeline
            import torch

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype= torch.bfloat16,
            )
            pipe = pipe.to(torch.bfloat16).to("cuda")

            pipe.set_progress_bar_config(disable=True)

            for prompt_idx, metadata in enumerate(tqdm(metadatas, desc="Generating images for prompts")):
                _, samples_dir = prepare_prompt_folder(output_dir, prompt_idx, metadata)
                for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_idx}", leave=False):
                    out_path = os.path.join(samples_dir, f"{seed:04d}.png")
                    if os.path.exists(out_path):
                        continue
                    try:
                        image = pipe(
                            metadata[key],
                            generator=torch.Generator(device="cuda").manual_seed(seed),
                        ).images[0]
                        image.save(out_path)
                    except Exception as e:
                        print(f"Error generating image for prompt {prompt_idx}, seed {seed}: {e}")

        case 'sd21':
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

            for prompt_idx, metadata in enumerate(tqdm(metadatas, desc="Generating images for prompts")):
                _, samples_dir = prepare_prompt_folder(output_dir, prompt_idx, metadata)

                for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_idx}", leave=False):
                    out_path = os.path.join(samples_dir, f"{seed:04d}.png")
                    if os.path.exists(out_path):
                        continue
                    image = pipe(metadata[key], generator=torch.manual_seed(seed)).images[0]
                    image.save(out_path)

        case 'comat21':
            from models.comat import CoMat
            import torch
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, 'configs/comat/model/sd21.yaml'))
            model_cfg.do_classifier_free_guidance = True
            
            try:
                model = CoMat(**model_cfg).to(dtype=torch.bfloat16).cuda()
            except Exception as e:
                model_cfg.pretrain_model = "hanker917/stable-diffusion-2-1"
                model = CoMat(**model_cfg).to(dtype=torch.bfloat16).cuda()
            model.load_checkpoint(os.path.join(PROJECT_ROOT, "checkpoints/CoMat/comat21/step=2000-g_loss=6.6597-d_loss=0.3280.ckpt"))            
            model.pipeline.set_progress_bar_config(disable=True)
            
            for prompt_idx, metadata in enumerate(tqdm(metadatas, desc="Generating images for prompts")):
                _, samples_dir = prepare_prompt_folder(output_dir, prompt_idx, metadata)

                for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_idx}", leave=False):
                    out_path = os.path.join(samples_dir, f"{seed:04d}.png")
                    if os.path.exists(out_path):
                        continue
                    image = model.predict(
                        metadata[key],
                        height=768,
                        width=768,
                        generator=torch.manual_seed(seed),
                    ).images[0]
                    image.save(out_path)
        case 'flux1':
            import torch
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16,
            ).to(torch.bfloat16).to("cuda")

            pipe.set_progress_bar_config(disable=True)

            for prompt_idx, metadata in enumerate(tqdm(metadatas, desc="Generating images for prompts")):
                _, samples_dir = prepare_prompt_folder(output_dir, prompt_idx, metadata)

                for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_idx}", leave=False):
                    out_path = os.path.join(samples_dir, f"{seed:04d}.png")
                    if os.path.exists(out_path):
                        continue

                    with torch.inference_mode():
                        image = pipe(
                            metadata[key],
                            guidance_scale=0.0,
                            num_inference_steps=4,
                            max_sequence_length=256,
                            generator=torch.Generator(device="cuda").manual_seed(seed),
                        ).images[0]
                    image.save(out_path)

        case 'flux.2':
            import torch
            from diffusers import Flux2Pipeline, AutoModel
            from transformers import Mistral3ForConditionalGeneration
            from diffusers.utils import load_image

            repo_id = "diffusers/FLUX.2-dev-bnb-4bit" #quantized text-encoder and DiT. VAE still in bf16
            device = "cuda:0"
            dtype = torch.bfloat16

            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                repo_id, subfolder="text_encoder", dtype=torch.bfloat16, device_map="cpu"
            )
            dit = AutoModel.from_pretrained(
                repo_id, subfolder="transformer", dtype=torch.bfloat16, device_map="cpu"
            )
            pipe = Flux2Pipeline.from_pretrained(
                repo_id, text_encoder=text_encoder, transformer=dit, dtype=dtype
            )
            pipe = pipe.to(device)
            # pipe.enable_model_cpu_offload()

            prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
            #cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
            image = pipe(
                prompt=prompt,
                #image=[cat_image] #multi-image input
                generator=torch.Generator(device=device).manual_seed(42),
                num_inference_steps=50,
                guidance_scale=4,
            ).images[0]

            image.save("flux2_output.png")

        case 'pixart-sigma':
            import torch
            from diffusers import Transformer2DModel, PixArtSigmaPipeline

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            weight_dtype = torch.bfloat16

            pipe = PixArtSigmaPipeline.from_pretrained(
                "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
                torch_dtype=weight_dtype,
                use_safetensors=True,
            )
            pipe.to(weight_dtype).to(device)

            pipe.set_progress_bar_config(disable=True)

            # Enable memory optimizations.
            # pipe.enable_model_cpu_offload()


            for prompt_idx, metadata in enumerate(tqdm(metadatas, desc="Generating images for prompts")):
                _, samples_dir = prepare_prompt_folder(output_dir, prompt_idx, metadata)

                for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_idx}", leave=False):
                    out_path = os.path.join(samples_dir, f"{seed:04d}.png")
                    if os.path.exists(out_path):
                        continue
                    image = pipe(
                        metadata[key],
                        generator=torch.Generator(device=device).manual_seed(seed),
                        ).images[0]
                    image.save(out_path)

        case _:
            raise ValueError(f"Unsupported model name: {model_name}")
        
def generate_images_for_PAE(
        pae_model_name: str,
        metadatas: list,
        output_dir: str,
        images_per_prompt: int = 4,
        user_refiner: bool = False,
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    key = 'prompt' if not user_refiner else 'refined_prompt'
    print(f"Using key '{key}' for prompts.")

    match pae_model_name:
        case 'pae_sd21':
            from diffusers import UniPCMultistepScheduler
            from models.utils.PAE.dynamicpipeline import StableDiffusionDynamicPromptPipeline
            import torch
            try:
                pipe = StableDiffusionDynamicPromptPipeline.from_pretrained(
                    "stabilityai/stable-diffusion-2-1",
                    dtype=torch.bfloat16,
                )
            except Exception as e:
                pipe = StableDiffusionDynamicPromptPipeline.from_pretrained(
                    "hanker917/stable-diffusion-2-1",
                    dtype=torch.bfloat16,
                )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")
            pipe.set_progress_bar_config(disable=True)
            for prompt_idx, metadata in enumerate(tqdm(metadatas, desc="Generating images for prompts")):
                _, samples_dir = prepare_prompt_folder(output_dir, prompt_idx, metadata)
                for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_idx}", leave=False):
                    out_path = os.path.join(samples_dir, f"{seed:04d}.png")
                    if os.path.exists(out_path):
                        continue
                    image = pipe(metadata[key], generator=torch.manual_seed(seed), num_inference_steps=10).images[0]
                    image.save(out_path)

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate images using specified text-to-image model.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to use (e.g., sd3, sd21)')
    parser.add_argument('--refiner_type', type=str, default=None, help='Type of refiner to use for prompts (optional)')
    args = parser.parse_args()
    if args.refiner_type:
        data = load_geneval_data(os.path.join(PROJECT_ROOT, f'data/gen_eval_data/refined_prompts/{args.refiner_type}_refined_prompts.jsonl'))
        output_dir = f'data/gen_eval_data/{args.refiner_type}/{args.model}'
        print(f"Loaded {len(data)} refined prompts using refiner type: {args.refiner_type}")
        print(f"Output directory: {output_dir}")
    else:
        data = load_geneval_data(os.path.join(PROJECT_ROOT, 'data/gen_eval_data/evaluation_metadata.jsonl'))
        output_dir = f'data/gen_eval_data/original/{args.model}'
        print(f"Loaded {len(data)} original prompts.")
        print(f"Output directory: {output_dir}")



    if 'pae' in args.model:
        generate_images_for_PAE(
            pae_model_name=args.model,
            metadatas=data,
            user_refiner=bool(args.refiner_type),
            output_dir=f'data/gen_eval_data/PAE/{args.model.split("_")[-1]}',
            images_per_prompt=4,
        )
    else:
        generate_images(
            model_name=args.model,
            metadatas=data,
            user_refiner=bool(args.refiner_type),
            output_dir=output_dir,
            images_per_prompt=4,
        )