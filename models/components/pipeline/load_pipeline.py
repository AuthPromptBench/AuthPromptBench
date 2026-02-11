from typing import Union
import torch

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler

from .AttrConcenTrainableSDPipeline import AttrConcenTrainableSDPipeline
from .TrainableSDPipeline import TrainableSDPipeline

from peft import (
    LoraConfig,
    get_peft_model,
)

def _load_diffusion_pipeline(
        model_path: str, 
        model_name: str, 
        revision: str, 
        weight_dtype: torch.dtype,
        ):
    """
    Load the diffusion pipeline based on the model name and path.

    Args:
        model_path (str): Path to the pretrained model.
        model_name (str): Name of the model to load.
        revision (str): Revision of the model to load.
        weight_dtype (torch.dtype): Data type for the model weights.
    """

    if model_name == 'sd_1_5':
        pipeline = TrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_dtype=weight_dtype)
    elif model_name == 'sd_1_5_attrcon':
        pipeline = AttrConcenTrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_dtype=weight_dtype)
    elif model_name == 'sd_2_1':
        pipeline = TrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_dtype=weight_dtype)
    elif model_name == 'sd_2_1_attrcon':
        pipeline = AttrConcenTrainableSDPipeline.from_pretrained(model_path, revision=revision, torch_dtype=weight_dtype)
    elif 'sdxl' in model_name:
        # TODO: support sdxl unet
        raise NotImplementedError("SDXL unet is not supported yet")
        vae_path = "madebyollin/sdxl-vae-fp16-fix"
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=torch.float16)
        if 'unet' in model_name:
            unet = UNet2DConditionModel.from_pretrained(args.sdxl_unet_path, revision=revision)
        else:
            unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", revision=revision)
        if 'attrcon' in model_name:
            PIPELINE_NAME = AttrConcenTrainableSDXLPipeline
        else:
            PIPELINE_NAME = TrainableSDXLPipeline
        pipeline = PIPELINE_NAME.from_pretrained(model_path, revision=revision, vae=vae, unet=unet, torch_dtype=weight_dtype)
            
    else:
        raise NotImplementedError("This model is not supported yet")
    return pipeline


def load_pipeline(
        pretrain_model: str, 
        model_name: str, 
        weight_dtype: torch.dtype, 
        full_finetuning: bool = False,
        tune_vae: bool = False,
        tune_text_encoder: bool = False,
        lora_rank: int = 128,
        train_text_encoder_lora: bool = False,
        is_D=False,
        revision: str = None,
        scheduler: str = "DDPM",
        gradient_checkpointing: bool = False,
        ):
    """
    Load the diffusion pipeline with the specified configurations.

    Args:
        pretrain_model (str): Path to the pretrained model.
        model_name (str): Name of the model to load. Should be one of 'sd_1_5', 'sd_1_5_attrcon', or 'sdxl'.
        weight_dtype (torch.dtype): Data type for the model weights.
        full_finetuning (bool): Whether to perform full finetuning unet. If False, LoRA will be used.
        tune_vae (bool): Whether to tune the VAE.
        tune_text_encoder (bool): Whether to tune the text encoder.
        lora_rank (int): Rank for LoRA.
        train_text_encoder_lora (bool): Whether to train the text encoder with LoRA.
        is_D (bool): Whether the pipeline is for discriminator.
        revision (str): Revision of the model to load.
        scheduler (str): Scheduler type to use ("DPM++" or "DDPM").
        gradient_checkpointing (bool): Whether to enable gradient checkpointing.
    Returns:
        Union[TrainableSDPipeline, AttrConcenTrainableSDPipeline]: The loaded pipeline
    """
    # Load pipeline
    pipeline = _load_diffusion_pipeline(pretrain_model, model_name, revision, weight_dtype)

    scheduler_args = {}

    if scheduler == "DPM++":
         pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DDPM":
        if "variance_type" in  pipeline.scheduler.config:
            variance_type =  pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    if full_finetuning:
         pipeline.unet.to(dtype=torch.float)
    else:
        pipeline.unet.to(dtype=weight_dtype)
    pipeline.vae.to(dtype=weight_dtype)
    pipeline.text_encoder.to(dtype=weight_dtype)
    # set grad
    # Freeze vae and text_encoder
    pipeline.vae.requires_grad_(tune_vae)
    pipeline.text_encoder.requires_grad_(tune_text_encoder)
    pipeline.unet.requires_grad_(False)
    if hasattr(pipeline, "safety_checker") and pipeline.safety_checker is not None:
        # Freeze safety checker if it exists
        pipeline.safety_checker.requires_grad_(False)

    # gradient checkpoint
    if gradient_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()
        # TODO: check
        if tune_text_encoder or train_text_encoder_lora:
            pipeline.text_encoder.gradient_checkpointing_enable()
            pipeline.text_encoder_2.gradient_checkpointing_enable() if hasattr(pipeline, "text_encoder_2") else None
    
    # set trainable lora
    pipeline = set_pipeline_trainable_module(pipeline=pipeline, is_D=is_D, full_finetuning=full_finetuning,
                                             lora_rank=lora_rank, train_text_encoder_lora=train_text_encoder_lora)
    
    return pipeline

def set_pipeline_trainable_module( 
        pipeline: Union[TrainableSDPipeline, AttrConcenTrainableSDPipeline], 
        is_D: bool = False,
        full_finetuning: bool = False,
        lora_rank: int = 128,
        train_text_encoder_lora: bool = False,
        ):
    """
    Set the trainable modules in the pipeline.

    Args:
        pipeline (Union[TrainableSDPipeline, AttrConcenTrainableSDPipeline]): The pipeline to modify.
        is_D (bool): Whether the pipeline is for discriminator.
        full_finetuning (bool): Whether to perform full finetuning.
        lora_rank (int): The rank for LoRA.
        train_text_encoder_lora (bool): Whether to train the text encoder with LoRA.

    Returns:
        Union[TrainableSDPipeline, AttrConcenTrainableSDPipeline]: The modified pipeline
    """
    if not full_finetuning:
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha= lora_rank,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    if train_text_encoder_lora and not is_D:
        # ensure that dtype is float32, even if rest of the model that isn't trained is loaded in fp16
        # text_lora_parameters = LoraLoaderMixin._modify_text_encoder(pipeline.text_encoder, dtype=torch.float32, rank=args.lora_rank)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=0.0,
            bias="none",
        )
        pipeline.text_encoder = get_peft_model(pipeline.text_encoder, lora_config)

    return pipeline

def get_trainable_parameters(
        pipeline: Union[TrainableSDPipeline, AttrConcenTrainableSDPipeline],
        is_D: bool = False,
        full_finetuning: bool = False,
        tune_vae: bool = False,
        tune_text_encoder: bool = False,
        train_text_encoder_lora: bool = False,
        ):
    """
    Get the trainable parameters from the pipeline.

    Args:
        pipeline (Union[TrainableSDPipeline, AttrConcenTrainableSDPipeline]): The pipeline to get parameters from.
        is_D (bool): Whether the pipeline is for discriminator.
        full_finetuning (bool): Whether to perform full finetuning.
        tune_vae (bool): Whether to tune the VAE.
        tune_text_encoder (bool): Whether to tune the text encoder.
        train_text_encoder_lora (bool): Whether to train the text encoder with LoRA.
    Returns:
        tuple: A tuple containing the trainable parameters for G and text lora parameters.
    """
    # load unet parameters
    if full_finetuning:
        G_parameters = list(pipeline.unet.parameters())
    else:
        G_parameters = []
        # get lora parameters
        for name, param in pipeline.unet.named_parameters():
            if param.requires_grad:
                G_parameters.append(param)
    
    # load other parameters, not for D
    if not is_D:
        text_lora_parameters = []
        if tune_vae:
            G_parameters.extend(pipeline.vae.parameters())
        
        if tune_text_encoder:
            G_parameters.extend(pipeline.text_encoder.parameters())
        
        if train_text_encoder_lora:
            for n, p in pipeline.text_encoder.named_parameters():
                if 'lora' in n:
                    if not p.requires_grad:
                        import pdb
                        pdb.set_trace()
                    # TODO:当前配置默认float32
                    # if p.dtype != torch.float:
                    #     p.data = p.data.to(torch.float)
                    text_lora_parameters.append(p)
    
        return G_parameters, text_lora_parameters
    else:
        return G_parameters