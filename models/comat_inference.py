from typing import Any, Dict, List, Optional, Union

import torch

from .components.pipeline.load_pipeline import load_pipeline


class CoMatInference(torch.nn.Module):
    """Lightweight CoMat wrapper for image generation.

    This class intentionally does not import or initialize the training-only
    reward model, discriminator, segmentation model, or Lightning module.
    """

    def __init__(
        self,
        pretrain_model: str,
        model_name: str,
        weight_dtype: str,
        full_finetuning: bool,
        tune_vae: bool,
        tune_text_encoder: bool,
        train_text_encoder_lora: bool,
        lora_rank: int,
        revision: str,
        scheduler: str,
        gradient_checkpointing: bool,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        self.weight_dtype = self._resolve_dtype(weight_dtype)
        self.pipeline = load_pipeline(
            pretrain_model=pretrain_model,
            model_name=model_name,
            weight_dtype=self.weight_dtype,
            full_finetuning=full_finetuning,
            tune_vae=tune_vae,
            tune_text_encoder=tune_text_encoder,
            train_text_encoder_lora=train_text_encoder_lora,
            lora_rank=lora_rank,
            is_D=False,
            revision=revision,
            scheduler=scheduler,
            gradient_checkpointing=gradient_checkpointing,
        )
        backbone = {
            "unet": self.pipeline.unet,
            "vae": self.pipeline.vae,
            "text_encoder": self.pipeline.text_encoder,
        }
        if getattr(self.pipeline, "safety_checker", None) is not None:
            backbone["safety_checker"] = self.pipeline.safety_checker
        self.backbone = torch.nn.ModuleDict(backbone)

    @staticmethod
    def _resolve_dtype(weight_dtype: str) -> torch.dtype:
        if weight_dtype == "fp16":
            return torch.float16
        if weight_dtype == "fp32":
            return torch.float32
        if weight_dtype == "bf16":
            return torch.bfloat16
        raise ValueError(f"Unsupported weight_dtype: {weight_dtype}")

    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=False)
        print(f"Checkpoint loaded from {checkpoint_path}")

    @torch.no_grad()
    def predict(
        self,
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        return self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            **kwargs,
        )
