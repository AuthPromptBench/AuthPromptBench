import os
import json
from tqdm import tqdm
import math
import torch
import multiprocessing

from metrics.utils.load_benchmark import (
    DATA_FILE_PATH,
    PROJECT_ROOT,
    load_benchmark_data,
    load_refined_data
)

from huggingface_hub import login


login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")


def generate_images(
        model_name: str,
        prompts: list,
        output_dir: str,
        images_per_prompt: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        start_index: int = 0,
        end_index: int = None,
):
    """
    Generate images using the specified model and save them to the output directory.

    Parameters:
    - model_name: The name of the text-to-image generation model to use.
    - prompts: A list of text prompts.
    - output_dir: The directory path to save the generated images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    match model_name:
        case 'sd3':
            from diffusers import StableDiffusion3Pipeline
            import torch

            # torch.set_float32_matmul_precision("high")

            # torch._inductor.config.conv_1x1_as_mm = True
            # torch._inductor.config.coordinate_descent_tuning = True
            # torch._inductor.config.epilogue_fusion = False
            # torch._inductor.config.coordinate_descent_check_all_directions = True

            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype= torch.bfloat16,
            )
            pipe = pipe.to(torch.bfloat16).to(device)

            pipe.set_progress_bar_config(disable=True)

            # pipe.transformer.to(memory_format=torch.channels_last)
            # pipe.vae.to(memory_format=torch.channels_last)

            # pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
            # pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

            # # Warm Up
            # for _ in range(3):
            #     _ = pipe(prompt="a photo of a cat holding a sign that says hello world", generator=torch.manual_seed(1))
            batch_size = 8
            if batch_size == 1:
                for idx, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
                    prompt_dir = os.path.join(output_dir, f"prompt_{idx+start_index}")
                    os.makedirs(prompt_dir, exist_ok=True)

                    for i in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {idx}", leave=False):
                        if os.path.exists(os.path.join(prompt_dir, f"{i}.png")):
                            continue
                        image = pipe(prompt, generator=torch.manual_seed(i)).images[0]
                        image.save(os.path.join(prompt_dir, f"{i}.png"))
            else:
                for i in tqdm(range(math.ceil(len(prompts)/batch_size)), desc="Generating images in batches"):
                    batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
                    for idx in tqdm(range(8), desc='Seeds', leave=False):
                        # 检查是否已经生成过
                        skip_batch = True
                        for j in range(len(batch_prompts)):
                            prompt_idx = i * batch_size + j
                            prompt_path = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}", f"{idx}.png")
                            if not os.path.exists(prompt_path):
                                skip_batch = False
                                break
                        if skip_batch:
                            continue
                        images = pipe(batch_prompts, generator=torch.manual_seed(idx)).images
                        for j, image in enumerate(images):
                            prompt_idx = i * batch_size + j
                            prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}")
                            os.makedirs(prompt_dir, exist_ok=True)
                            image.save(os.path.join(prompt_dir, f"{idx}.png"))

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
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)

            batch_size = 4
            if batch_size == 1:
                for idx, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
                    prompt_dir = os.path.join(output_dir, f"prompt_{idx+start_index}")
                    os.makedirs(prompt_dir, exist_ok=True)

                    for i in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {idx}", leave=False):
                        if os.path.exists(os.path.join(prompt_dir, f"{i}.png")):
                            continue
                        image = pipe(prompt, generator=torch.manual_seed(i)).images[0]
                        image.save(os.path.join(prompt_dir, f"{i}.png"))
            else:
                for i in tqdm(range(math.ceil(len(prompts)/batch_size)), desc="Generating images in batches"):
                    batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
                    for idx in tqdm(range(8), desc='Seeds', leave=False):
                        # 检查是否已经生成过
                        skip_batch = True
                        for j in range(len(batch_prompts)):
                            prompt_idx = i * batch_size + j
                            prompt_path = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}", f"{idx}.png")
                            if not os.path.exists(prompt_path):
                                skip_batch = False
                                break
                        if skip_batch:
                            continue
                        images = pipe(batch_prompts, generator=torch.manual_seed(idx)).images
                        for j, image in enumerate(images):
                            prompt_idx = i * batch_size + j
                            prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}")
                            os.makedirs(prompt_dir, exist_ok=True)
                            image.save(os.path.join(prompt_dir, f"{idx}.png"))

        case 'comat21':
            from models.comat import CoMat
            import torch
            from omegaconf import OmegaConf
            model_cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, 'configs/comat/model/sd21.yaml'))
            model_cfg.do_classifier_free_guidance = True
            
            try:
                model = CoMat(**model_cfg).to(dtype=torch.bfloat16).to(device)
            except Exception as e:
                model_cfg.pretrain_model = "hanker917/stable-diffusion-2-1"
                model = CoMat(**model_cfg).to(dtype=torch.bfloat16).to(device)
            model.load_checkpoint(os.path.join(PROJECT_ROOT, "checkpoints/CoMat/comat21/step=2000-g_loss=6.6597-d_loss=0.3280.ckpt"))            
            model.pipeline.set_progress_bar_config(disable=True)
            
            for idx, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
                prompt_dir = os.path.join(output_dir, f"prompt_{idx+start_index}")
                os.makedirs(prompt_dir, exist_ok=True)

                for i in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {idx}", leave=False):
                    if os.path.exists(os.path.join(prompt_dir, f"{i}.png")):
                        continue
                    image = model.predict(
                        prompt,
                        height=768,
                        width=768,
                        generator=torch.manual_seed(i),
                    ).images[0]
                    image.save(os.path.join(prompt_dir, f"{i}.png"))
        case 'flux1':
            import torch
            from diffusers import FluxPipeline

            pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-schnell",
                torch_dtype=torch.bfloat16,
            ).to(torch.bfloat16).to(device)

            pipe.set_progress_bar_config(disable=True)

            batch_size = 1

            if batch_size == 1:
                for idx, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
                    prompt_dir = os.path.join(output_dir, f"prompt_{idx+start_index}")
                    os.makedirs(prompt_dir, exist_ok=True)

                    for i in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {idx}", leave=False):
                        if os.path.exists(os.path.join(prompt_dir, f"{i}.png")):
                            continue

                        with torch.inference_mode():
                            image = pipe(
                                prompt,
                                guidance_scale=0.0,
                                num_inference_steps=4,
                                max_sequence_length=256,
                                generator=torch.Generator(device="cuda").manual_seed(i),
                            ).images[0]
                        image.save(os.path.join(prompt_dir, f"{i}.png"))
            else:
                for i in tqdm(range(math.ceil(len(prompts)/batch_size)), desc="Generating images in batches"):
                    batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
                    for idx in tqdm(range(8), desc='Seeds', leave=False):
                        # 检查是否已经生成过
                        skip_batch = True
                        for j in range(len(batch_prompts)):
                            prompt_idx = i * batch_size + j
                            prompt_path = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}", f"{idx}.png")
                            if not os.path.exists(prompt_path):
                                skip_batch = False
                                break
                        if skip_batch:
                            continue
                        with torch.inference_mode():
                            images = pipe(
                                batch_prompts,
                                guidance_scale=0.0,
                                num_inference_steps=4,
                                max_sequence_length=256,
                                generator=torch.Generator(device="cuda").manual_seed(idx),
                            ).images
                        for j, image in enumerate(images):
                            prompt_idx = i * batch_size + j
                            prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}")
                            os.makedirs(prompt_dir, exist_ok=True)
                            image.save(os.path.join(prompt_dir, f"{idx}.png"))


        case 'flux.2':
            import torch
            from diffusers import Flux2Pipeline, AutoModel
            from transformers import Mistral3ForConditionalGeneration
            from diffusers.utils import load_image

            repo_id = "diffusers/FLUX.2-dev-bnb-4bit" #quantized text-encoder and DiT. VAE still in bf16
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
            from diffusers import PixArtSigmaPipeline

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

            batch_size = 1
            if batch_size == 1:
                for idx, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
                    prompt_dir = os.path.join(output_dir, f"prompt_{idx+start_index}")
                    os.makedirs(prompt_dir, exist_ok=True)

                    for i in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {idx}", leave=False):
                        if os.path.exists(os.path.join(prompt_dir, f"{i}.png")):
                            continue
                        image = pipe(
                            prompt,
                            generator=torch.Generator(device=device).manual_seed(i),
                            ).images[0]
                        image.save(os.path.join(prompt_dir, f"{i}.png"))
            else:
                for i in tqdm(range(math.ceil(len(prompts)/batch_size)), desc="Generating images in batches"):
                    batch_prompts = prompts[i*batch_size:(i+1)*batch_size]
                    for idx in tqdm(range(8), desc='Seeds', leave=False):
                        # 检查是否已经生成过
                        skip_batch = True
                        for j in range(len(batch_prompts)):
                            prompt_idx = i * batch_size + j
                            prompt_path = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}", f"{idx}.png")
                            if not os.path.exists(prompt_path):
                                skip_batch = False
                                break
                        if skip_batch:
                            continue
                        images = pipe(
                            batch_prompts,
                            generator=torch.Generator(device=device).manual_seed(idx),
                            ).images
                        for j, image in enumerate(images):
                            prompt_idx = i * batch_size + j
                            prompt_dir = os.path.join(output_dir, f"prompt_{prompt_idx+start_index}")
                            os.makedirs(prompt_dir, exist_ok=True)
                            image.save(os.path.join(prompt_dir, f"{idx}.png"))

        case _:
            raise ValueError(f"Unsupported model name: {model_name}")
        
def generate_images_for_PAE(
        pae_model_name: str,
        prompts: list,
        output_dir: str,
        images_per_prompt: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        start_index: int = 0,
        end_index: int = None,
):
    """   
    Generate images using the specified PAE model and save them to the output directory.

    Parameters:
    - pae_model_name: The name of the PAE text-to-image generation model to use.
    - prompts: A list of text prompts.
    - output_dir: The directory path to save the generated images.
    - device: 设备类型，默认为 "cuda" 如果可用，否则为 "cpu"。
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            for idx, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
                prompt_dir = os.path.join(output_dir, f"prompt_{idx+start_index}")
                os.makedirs(prompt_dir, exist_ok=True)

                for i in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {idx}", leave=False):
                    if os.path.exists(os.path.join(prompt_dir, f"{i}.png")):
                        continue
                    image = pipe(prompt, generator=torch.manual_seed(i), num_inference_steps=10).images[0]
                    image.save(os.path.join(prompt_dir, f"{i}.png"))

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate images using specified text-to-image model.")
    parser.add_argument('--model', type=str, required=True, help='Name of the model to use (e.g., sd3, sd21)')
    parser.add_argument('--refiner_type', type=str, default=None, help='Type of refiner to use for prompts (optional)')
    parser.add_argument('--start_index', type=int, default=0, help='Starting index of prompts to process (optional)')
    parser.add_argument('--end_index', type=int, default=None, help='Ending index of prompts to process (optional)')
    args = parser.parse_args()
    if args.refiner_type:
        data = load_refined_data(args.refiner_type)
        output_dir = f'data/benchmark/generated_images/{args.refiner_type}/{args.model}'
        print(f"Loaded {len(data)} refined prompts using refiner type: {args.refiner_type}")
        print(f"Output directory: {output_dir}")
    else:
        data = load_benchmark_data()
        output_dir = f'data/benchmark/generated_images/{args.model}'
        print(f"Loaded {len(data)} original prompts.")
        print(f"Output directory: {output_dir}")


    if args.end_index is None:
        args.end_index = len(data)
    prompts = [item['prompt'] for item in data[args.start_index:args.end_index]]
    prompts = [p if p != None else "" for p in prompts]



    if 'pae' in args.model:
        generate_images_for_PAE(
            pae_model_name=args.model,
            prompts=prompts,
            output_dir=f'data/benchmark/generated_images/PAE/{args.model.split("_")[-1]}',
            images_per_prompt=8,
            start_index=args.start_index,
            end_index=args.end_index,
        )
    else:
        generate_images(
            model_name=args.model,
            prompts=prompts,
            output_dir=output_dir,
            images_per_prompt=8,
            start_index=args.start_index,
            end_index=args.end_index,
        )