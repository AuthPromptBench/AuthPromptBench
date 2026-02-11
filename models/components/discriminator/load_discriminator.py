import torch
from .gan_sdxl import D_sd #, D_sdxl


def load_discriminator(
        gan_model_arch: str, 
        weight_dtype: torch.dtype, 
        scheduler: str,
        full_finetuning: bool,
        gan_unet_lastlayer_cls: bool,
        condition_discriminator: bool,
        ):
    gan_model_arch = gan_model_arch.replace('gan_', '')
    
    if gan_model_arch == 'sd_1_5':
        return D_sd(gan_model_arch, weight_dtype, scheduler, full_finetuning, gan_unet_lastlayer_cls, condition_discriminator)
    elif gan_model_arch == 'sd_2_1':
        return D_sd(gan_model_arch, weight_dtype, scheduler, full_finetuning, gan_unet_lastlayer_cls, condition_discriminator)
    # elif 'sdxl' in gan_model_arch :
    #     return D_sdxl(args, weight_dtype, device)