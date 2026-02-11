import argparse
import math
import os
import sys
from types import SimpleNamespace
from typing import Callable, Iterable, List, Optional, Sequence

from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics.utils.load_benchmark import PROJECT_ROOT, load_benchmark_data, load_refined_data


DEFAULT_IMAGES_PER_PROMPT = 8


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def maybe_login_huggingface() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        return
    from huggingface_hub import login

    login(token=token)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def image_path(output_dir: str, prompt_index: int, seed: int) -> str:
    return os.path.join(output_dir, f"prompt_{prompt_index}", f"{seed}.png")


def missing_seed(output_dir: str, prompt_index: int, seed: int) -> bool:
    return not os.path.exists(image_path(output_dir, prompt_index, seed))


def save_image(image, output_dir: str, prompt_index: int, seed: int) -> None:
    prompt_dir = os.path.join(output_dir, f"prompt_{prompt_index}")
    ensure_dir(prompt_dir)
    image.save(os.path.join(prompt_dir, f"{seed}.png"))


def iter_batches(items: Sequence[str], batch_size: int) -> Iterable[tuple[int, Sequence[str]]]:
    for batch_index in range(math.ceil(len(items) / batch_size)):
        start = batch_index * batch_size
        yield start, items[start : start + batch_size]


def generate_single_prompt_images(
    prompts: Sequence[str],
    output_dir: str,
    generate_one: Callable[[str, int], object],
    images_per_prompt: int,
    start_index: int,
) -> None:
    for local_index, prompt in enumerate(tqdm(prompts, desc="Generating images for prompts")):
        prompt_index = local_index + start_index
        for seed in tqdm(range(images_per_prompt), desc=f"Generating images for prompt {prompt_index}", leave=False):
            if not missing_seed(output_dir, prompt_index, seed):
                continue
            save_image(generate_one(prompt, seed), output_dir, prompt_index, seed)


def generate_batched_prompt_images(
    prompts: Sequence[str],
    output_dir: str,
    generate_batch: Callable[[Sequence[str], int], Sequence[object]],
    images_per_prompt: int,
    batch_size: int,
    start_index: int,
) -> None:
    for batch_start, batch_prompts in tqdm(
        iter_batches(prompts, batch_size),
        total=math.ceil(len(prompts) / batch_size),
        desc="Generating images in batches",
    ):
        for seed in tqdm(range(images_per_prompt), desc="Seeds", leave=False):
            missing = [
                local_index
                for local_index in range(len(batch_prompts))
                if missing_seed(output_dir, start_index + batch_start + local_index, seed)
            ]
            if not missing:
                continue

            images = generate_batch(batch_prompts, seed)
            for local_index in missing:
                save_image(
                    images[local_index],
                    output_dir,
                    start_index + batch_start + local_index,
                    seed,
                )


def load_sd3(device: str):
    import torch
    from diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to(torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_sd21(device: str):
    import torch
    from diffusers import StableDiffusionPipeline

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(
            "hanker917/stable-diffusion-2-1",
            torch_dtype=torch.bfloat16,
        )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_comat21(device: str):
    import torch
    from omegaconf import OmegaConf

    from models.comat_inference import CoMatInference

    model_cfg = OmegaConf.load(os.path.join(PROJECT_ROOT, "configs/comat/model/sd21.yaml"))
    model_cfg.do_classifier_free_guidance = True
    try:
        model = CoMatInference(**model_cfg).to(dtype=torch.bfloat16).to(device)
    except Exception:
        model_cfg.pretrain_model = "hanker917/stable-diffusion-2-1"
        model = CoMatInference(**model_cfg).to(dtype=torch.bfloat16).to(device)
    model.load_checkpoint(
        os.path.join(PROJECT_ROOT, "checkpoints/CoMat/comat21/step=2000-g_loss=6.6597-d_loss=0.3280.ckpt")
    )
    model.pipeline.set_progress_bar_config(disable=True)
    return model


def load_flux1(device: str):
    import torch
    from diffusers import FluxPipeline

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.bfloat16,
    ).to(torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_pixart_sigma(device: str):
    import torch
    from diffusers import PixArtSigmaPipeline

    pipe = PixArtSigmaPipeline.from_pretrained(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )
    pipe.to(torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def resolve_infinity_repo_path() -> Optional[str]:
    candidates = [
        os.environ.get("INFINITY_REPO_PATH"),
        os.path.join(PROJECT_ROOT, "Infinity"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def infinity_weight_path(repo_path: str, env_name: str, default_relative_path: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    return os.path.join(repo_path, default_relative_path)


def parse_infinity_float_or_list(value: str):
    values = [float(item) for item in value.split(",")]
    if len(values) == 1:
        return values[0]
    return values


def resolve_showo2_repo_path() -> Optional[str]:
    candidates = [
        os.environ.get("SHOWO2_REPO_PATH"),
        os.path.join(PROJECT_ROOT, "show-o2"),
    ]
    for path in candidates:
        if path and os.path.isdir(path):
            return path
    return None


def showo2_path(repo_path: str, env_name: str, default_relative_path: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    return os.path.join(repo_path, default_relative_path)


def load_infi(device: str):
    if device != "cuda":
        raise RuntimeError("Infinity generation requires CUDA/HIP through torch.cuda.")

    repo_path = resolve_infinity_repo_path()
    if repo_path is None:
        raise RuntimeError(
            "Infinity repo not found. Set INFINITY_REPO_PATH to a checkout containing the infinity package."
        )
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    from infinity.models.infinity import Infinity
    from infinity.models.bsq_vae.vae import vae_model
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

    model_path = infinity_weight_path(repo_path, "INFINITY_MODEL_PATH", "weights/infinity_2b_reg.pth")
    vae_path = infinity_weight_path(repo_path, "INFINITY_VAE_PATH", "weights/infinity_vae_d32_reg.pth")
    text_encoder_name = os.environ.get("INFINITY_TEXT_ENCODER", "google/flan-t5-xl")

    tokenizer = AutoTokenizer.from_pretrained(text_encoder_name, revision=None, legacy=True)
    tokenizer.model_max_length = 512
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_name, torch_dtype=torch.float16)
    text_encoder.to(device)
    text_encoder.eval()
    text_encoder.requires_grad_(False)

    vae_type = int(os.environ.get("INFINITY_VAE_TYPE", "32"))
    codebook_size = 2**vae_type
    vae = vae_model(
        vae_path,
        "dynamic",
        vae_type,
        codebook_size,
        patch_size=16,
        encoder_ch_mult=[1, 2, 4, 4, 4],
        decoder_ch_mult=[1, 2, 4, 4, 4],
        test_mode=True,
    ).to(device)

    pn = os.environ.get("INFINITY_PN", "1M")
    scale_schedule = dynamic_resolution_h_w[1.0][pn]["scales"]
    scale_schedule = [(1, h, w) for _, h, w in scale_schedule]

    with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        model = Infinity(
            vae_local=vae,
            text_channels=2048,
            text_maxlen=512,
            shared_aln=True,
            raw_scale_schedule=None,
            checkpointing="full-block",
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=False,
            add_lvl_embeding_only_first_block=True,
            use_bit_label=True,
            rope2d_each_sa_layer=True,
            rope2d_normalized_by_hw=2,
            pn=pn,
            apply_spatial_patchify=False,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            depth=32,
            embed_dim=2048,
            num_heads=16,
            drop_path_rate=0.1,
            mlp_ratio=4,
            block_chunks=8,
        ).to(device=device)
        for block in model.unregistered_blocks:
            block.bfloat16()
        model.eval()
        model.requires_grad_(False)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.rng = torch.Generator(device=device)

    return SimpleNamespace(
        model=model,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scale_schedule=scale_schedule,
        pn=pn,
        cfg=parse_infinity_float_or_list(os.environ.get("INFINITY_CFG", "4")),
        tau=parse_infinity_float_or_list(os.environ.get("INFINITY_TAU", "0.5")),
        cfg_insertion_layer=int(os.environ.get("INFINITY_CFG_INSERTION_LAYER", "0")),
        vae_type=vae_type,
        sampling_per_bits=int(os.environ.get("INFINITY_SAMPLING_PER_BITS", "1")),
        enable_positive_prompt=os.environ.get("INFINITY_ENABLE_POSITIVE_PROMPT", "0") == "1",
    )


def load_showo2(device: str):
    if device != "cuda":
        raise RuntimeError("Show-O2 generation requires CUDA/HIP through torch.cuda.")

    repo_path = resolve_showo2_repo_path()
    if repo_path is None:
        raise RuntimeError("Show-O2 repo not found. Set SHOWO2_REPO_PATH to a show-o2 checkout.")
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    import torch
    from omegaconf import OmegaConf

    from models import Showo2Qwen2_5, WanVAE, omni_attn_mask_naive
    from models.misc import get_text_tokenizer, prepare_gen_input
    from transport import Sampler, create_transport
    from utils import denorm, get_hyper_params, load_state_dict, path_to_llm_name

    config_path = showo2_path(repo_path, "SHOWO2_CONFIG_PATH", "configs/showo2_1.5b_demo_512x512.yaml")
    config = OmegaConf.load(config_path)

    if config.model.weight_type == "bfloat16":
        weight_type = torch.bfloat16
    elif config.model.weight_type == "float32":
        weight_type = torch.float32
    else:
        raise NotImplementedError(f"Unsupported Show-O2 weight type: {config.model.weight_type}")

    if config.model.vae_model.type != "wan21":
        raise NotImplementedError(f"Unsupported Show-O2 VAE type: {config.model.vae_model.type}")
    vae_path = showo2_path(repo_path, "SHOWO2_VAE_PATH", config.model.vae_model.pretrained_model_path)
    vae_model = WanVAE(vae_pth=vae_path, dtype=weight_type, device=torch.device(device))

    llm_name = path_to_llm_name[config.model.showo.llm_model_path]
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=llm_name,
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model_path = os.environ.get("SHOWO2_MODEL_PATH", config.model.showo.pretrained_model_path)
        model = Showo2Qwen2_5.from_pretrained(model_path, use_safetensors=False).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict_path = showo2_path(repo_path, "SHOWO2_MODEL_PATH", config.model_path)
        model.load_state_dict(load_state_dict(state_dict_path))
    model.to(weight_type)
    model.eval()

    if config.model.showo.add_time_embeds:
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    (
        num_t2i_image_tokens,
        _num_mmu_image_tokens,
        _num_video_tokens,
        max_seq_len,
        max_text_len,
        image_latent_dim,
        patch_size,
        latent_width,
        latent_height,
        pad_id,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        _bov_id,
        _eov_id,
        img_pad_id,
        _vid_pad_id,
        guidance_scale,
    ) = get_hyper_params(config, text_tokenizer, showo_token_ids)

    guidance_scale = float(os.environ.get("SHOWO2_GUIDANCE_SCALE", "7.5"))
    config.transport.num_inference_steps = int(os.environ.get("SHOWO2_NUM_INFERENCE_STEPS", "50"))

    transport = create_transport(
        path_type=config.transport.path_type,
        prediction=config.transport.prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=config.transport.train_eps,
        sample_eps=config.transport.sample_eps,
        snr_type=config.transport.snr_type,
        do_shift=config.transport.do_shift,
        seq_len=num_t2i_image_tokens,
    )
    sampler = Sampler(transport)

    return SimpleNamespace(
        model=model,
        vae_model=vae_model,
        tokenizer=text_tokenizer,
        prepare_gen_input=prepare_gen_input,
        omni_attn_mask_naive=omni_attn_mask_naive,
        denorm=denorm,
        config=config,
        weight_type=weight_type,
        num_t2i_image_tokens=num_t2i_image_tokens,
        max_seq_len=max_seq_len,
        max_text_len=max_text_len,
        image_latent_dim=image_latent_dim,
        patch_size=patch_size,
        latent_width=latent_width,
        latent_height=latent_height,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        boi_id=boi_id,
        eoi_id=eoi_id,
        img_pad_id=img_pad_id,
        guidance_scale=guidance_scale,
        sampler=sampler,
        repo_path=repo_path,
    )


def load_pae_sd21(device: str):
    import torch
    from diffusers import UniPCMultistepScheduler

    from models.utils.PAE.dynamicpipeline import StableDiffusionDynamicPromptPipeline

    try:
        pipe = StableDiffusionDynamicPromptPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1",
            dtype=torch.bfloat16,
        )
    except Exception:
        pipe = StableDiffusionDynamicPromptPipeline.from_pretrained(
            "hanker917/stable-diffusion-2-1",
            dtype=torch.bfloat16,
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generator_for_device(device: str, seed: int):
    import torch

    if device == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def encode_infinity_prompt(tokenizer, text_encoder, prompts: Sequence[str], enable_positive_prompt: bool):
    import torch
    import torch.nn.functional as F

    if enable_positive_prompt:
        prompts = [augment_infinity_positive_prompt(prompt) for prompt in prompts]
    tokens = tokenizer(text=list(prompts), max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    with torch.no_grad():
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)["last_hidden_state"].float()
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    text_cond = []
    for length, features in zip(lens, text_features.unbind(0)):
        text_cond.append(features[:length])
    return torch.cat(text_cond, dim=0), lens, cu_seqlens_k, max(lens)


def augment_infinity_positive_prompt(prompt: str) -> str:
    person_terms = [
        "man",
        "woman",
        "men",
        "women",
        "boy",
        "girl",
        "child",
        "person",
        "human",
        "adult",
        "teenager",
        "employee",
        "employer",
        "worker",
        "mother",
        "father",
        "sister",
        "brother",
        "grandmother",
        "grandfather",
        "son",
        "daughter",
    ]
    if any(term in prompt for term in person_terms):
        return prompt + ". very smooth faces, good looking faces, face to the camera, perfect facial features"
    return prompt


def infinity_tensor_to_pil(image):
    import numpy as np
    from PIL import Image

    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[-1] == 3:
        # The reference script saves with cv2.imwrite, so the tensor is treated as BGR.
        image = image[..., ::-1]
    return Image.fromarray(image)


def generate_infi_batch(infi, prompts: Sequence[str], seed: int):
    import torch

    cfg_list = infi.cfg
    tau_list = infi.tau
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(infi.scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(infi.scale_schedule)
    text_cond_tuple = encode_infinity_prompt(
        infi.tokenizer,
        infi.text_encoder,
        prompts,
        infi.enable_positive_prompt,
    )
    cfg_insertion_layer = [infi.cfg_insertion_layer]
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            _, _, images = infi.model.autoregressive_infer_cfg(
                vae=infi.vae,
                scale_schedule=infi.scale_schedule,
                label_B_or_BLT=text_cond_tuple,
                g_seed=seed,
                B=len(prompts),
                negative_label_B_or_BLT=None,
                force_gt_Bhw=None,
                cfg_sc=3,
                cfg_list=cfg_list,
                tau_list=tau_list,
                top_k=900,
                top_p=0.97,
                returns_vemb=1,
                ratio_Bl1=None,
                gumbel=0,
                norm_cfg=False,
                cfg_exp_k=0.0,
                cfg_insertion_layer=cfg_insertion_layer,
                vae_type=infi.vae_type,
                softmax_merge_topk=-1,
                ret_img=True,
                trunk_scale=1000,
                gt_leak=0,
                gt_ls_Bl=None,
                inference_mode=True,
                sampling_per_bits=infi.sampling_per_bits,
            )
    return [infinity_tensor_to_pil(image) for image in images]


def generate_showo2_batch(showo2, prompts: Sequence[str], seed: int):
    import torch
    from PIL import Image

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_text_tokens, batch_text_tokens_null, batch_modality_positions, batch_modality_positions_null = (
        showo2.prepare_gen_input(
            prompts,
            showo2.tokenizer,
            showo2.num_t2i_image_tokens,
            showo2.bos_id,
            showo2.eos_id,
            showo2.boi_id,
            showo2.eoi_id,
            showo2.pad_id,
            showo2.img_pad_id,
            showo2.max_text_len,
            "cuda",
        )
    )

    z = torch.randn(
        (
            len(prompts),
            showo2.image_latent_dim,
            showo2.latent_height * showo2.patch_size,
            showo2.latent_width * showo2.patch_size,
        ),
        device="cuda",
        dtype=torch.bfloat16,
    )

    if showo2.guidance_scale > 0:
        z = torch.cat([z, z], dim=0)
        text_tokens = torch.cat([batch_text_tokens, batch_text_tokens_null], dim=0)
        modality_positions = torch.cat([batch_modality_positions, batch_modality_positions_null], dim=0)
    else:
        text_tokens = batch_text_tokens
        modality_positions = batch_modality_positions

    block_mask = showo2.omni_attn_mask_naive(
        text_tokens.size(0),
        showo2.max_seq_len,
        modality_positions,
        "cuda",
    ).to(showo2.weight_type)

    model_kwargs = dict(
        text_tokens=text_tokens,
        attention_mask=block_mask,
        modality_positions=modality_positions,
        output_hidden_states=True,
        max_seq_len=showo2.max_seq_len,
        guidance_scale=showo2.guidance_scale,
    )

    sample_fn = showo2.sampler.sample_ode(
        sampling_method=showo2.config.transport.sampling_method,
        num_steps=showo2.config.transport.num_inference_steps,
        atol=showo2.config.transport.atol,
        rtol=showo2.config.transport.rtol,
        reverse=showo2.config.transport.reverse,
        time_shifting_factor=showo2.config.transport.time_shifting_factor,
    )
    with torch.inference_mode():
        samples = sample_fn(z, showo2.model.t2i_generate, **model_kwargs)[-1]
        if showo2.guidance_scale > 0:
            samples = torch.chunk(samples, 2)[0]

        if showo2.config.model.vae_model.type == "wan21":
            samples = samples.unsqueeze(2)
            images = showo2.vae_model.batch_decode(samples)
            images = images.squeeze(2)
        else:
            raise NotImplementedError(f"Unsupported Show-O2 VAE type: {showo2.config.model.vae_model.type}")

    images = showo2.denorm(images)
    return [Image.fromarray(image) for image in images]


def generate_images(
    model_name: str,
    prompts: List[str],
    output_dir: str,
    images_per_prompt: int = DEFAULT_IMAGES_PER_PROMPT,
    device: Optional[str] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> None:
    del end_index
    import torch

    device = device or get_device()
    ensure_dir(output_dir)

    if model_name == "sd3":
        pipe = load_sd3(device)
        generate_batched_prompt_images(
            prompts, output_dir,
            lambda batch, seed: pipe(batch, generator=torch.manual_seed(seed)).images,
            images_per_prompt, batch_size=8, start_index=start_index,
        )
    elif model_name == "sd21":
        pipe = load_sd21(device)
        generate_batched_prompt_images(
            prompts, output_dir,
            lambda batch, seed: pipe(batch, generator=torch.manual_seed(seed)).images,
            images_per_prompt, batch_size=4, start_index=start_index,
        )
    elif model_name == "comat21":
        model = load_comat21(device)

        def generate_one(prompt: str, seed: int):
            return model.predict(
                prompt, height=768, width=768,
                generator=generator_for_device(device, seed),
            ).images[0]

        generate_single_prompt_images(prompts, output_dir, generate_one, images_per_prompt, start_index)
    elif model_name == "flux1":
        pipe = load_flux1(device)

        def generate_one(prompt: str, seed: int):
            with torch.inference_mode():
                return pipe(
                    prompt,
                    guidance_scale=0.0,
                    num_inference_steps=4,
                    max_sequence_length=256,
                    generator=generator_for_device(device, seed),
                ).images[0]

        generate_single_prompt_images(prompts, output_dir, generate_one, images_per_prompt, start_index)
    elif model_name == "pixart-sigma":
        pipe = load_pixart_sigma(device)
        generate_single_prompt_images(
            prompts, output_dir,
            lambda prompt, seed: pipe(prompt, generator=generator_for_device(device, seed)).images[0],
            images_per_prompt, start_index,
        )
    elif model_name == "infi":
        infi = load_infi(device)
        generate_batched_prompt_images(
            prompts, output_dir,
            lambda batch, seed: generate_infi_batch(infi, batch, seed),
            images_per_prompt, batch_size=3, start_index=start_index,
        )
    elif model_name == "showo2":
        showo2 = load_showo2(device)
        generate_batched_prompt_images(
            prompts, output_dir,
            lambda batch, seed: generate_showo2_batch(showo2, batch, seed),
            images_per_prompt,
            batch_size=int(os.environ.get("SHOWO2_BATCH_SIZE", "4")),
            start_index=start_index,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def generate_images_for_pae(
    pae_model_name: str,
    prompts: List[str],
    output_dir: str,
    images_per_prompt: int = DEFAULT_IMAGES_PER_PROMPT,
    device: Optional[str] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> None:
    del end_index
    import torch

    device = device or get_device()
    ensure_dir(output_dir)

    if pae_model_name != "pae_sd21":
        raise ValueError(f"Unsupported PAE model name: {pae_model_name}")
    pipe = load_pae_sd21(device)
    generate_single_prompt_images(
        prompts,
        output_dir,
        lambda prompt, seed: pipe(
            prompt,
            generator=generator_for_device(device, seed),
            num_inference_steps=10,
        ).images[0],
        images_per_prompt,
        start_index,
    )


def load_prompts(refiner_type: Optional[str], start_index: int, end_index: Optional[int]) -> tuple[List[str], str]:
    if refiner_type:
        data = load_refined_data(refiner_type)
        source = f"refined prompts using refiner type: {refiner_type}"
    else:
        data = load_benchmark_data()
        source = "original prompts"

    if end_index is None:
        end_index = len(data)
    prompts = [(item.get("prompt") or "") for item in data[start_index:end_index]]
    return prompts, source


def output_dir_for(model: str, refiner_type: Optional[str]) -> str:
    if "pae" in model:
        return f"data/benchmark/generated_images/PAE/{model.split('_')[-1]}"
    if refiner_type:
        return f"data/benchmark/generated_images/{refiner_type}/{model}"
    return f"data/benchmark/generated_images/{model}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark images using a text-to-image model.")
    parser.add_argument("--model", type=str, required=True, help="Model name, for example sd3, sd21, comat21, infi.")
    parser.add_argument("--refiner_type", type=str, default=None, help="Optional refined prompt source.")
    parser.add_argument("--start_index", type=int, default=0, help="First prompt index to process.")
    parser.add_argument("--end_index", type=int, default=None, help="Exclusive prompt end index.")
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=DEFAULT_IMAGES_PER_PROMPT,
        help=f"Images to generate per prompt, default {DEFAULT_IMAGES_PER_PROMPT}.",
    )
    args = parser.parse_args()

    maybe_login_huggingface()
    prompts, source = load_prompts(args.refiner_type, args.start_index, args.end_index)
    output_dir = output_dir_for(args.model, args.refiner_type)
    print(f"Loaded {len(prompts)} {source}.")
    print(f"Output directory: {output_dir}")

    if "pae" in args.model:
        generate_images_for_pae(
            pae_model_name=args.model,
            prompts=prompts,
            output_dir=output_dir,
            images_per_prompt=args.images_per_prompt,
            start_index=args.start_index,
            end_index=args.end_index,
        )
    else:
        generate_images(
            model_name=args.model,
            prompts=prompts,
            output_dir=output_dir,
            images_per_prompt=args.images_per_prompt,
            start_index=args.start_index,
            end_index=args.end_index,
        )


if __name__ == "__main__":
    main()
