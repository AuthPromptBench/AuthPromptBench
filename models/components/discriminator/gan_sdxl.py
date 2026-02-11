import torch
from torch import nn
import copy

from ..pipeline.load_pipeline import (
    load_pipeline,
    TrainableSDPipeline,
)
from ..pipeline.load_pipeline import get_trainable_parameters as _get_trainable_parameters

class D_sd(nn.Module):
    def __init__(
            self,
            gan_model_arch: str,
            weight_dtype: torch.dtype, 
            scheduler: str,
            full_finetuning: bool,
            gan_unet_lastlayer_cls: bool,
            condition_discriminator: bool,
            ) -> None:
        """
        Discriminator for GAN training with Stable Diffusion.

        Args:
            gan_model_arch (str): The architecture of the GAN model.
            weight_dtype (torch.dtype): The data type for the model weights.
            scheduler (str): The scheduler to use for the GAN model. Scheduler type to use ("DPM++" or "DDPM").
            full_finetuning (bool): Whether to perform full finetuning on the UNet. If False, LoRA will be used.
            gan_unet_lastlayer_cls (bool): Whether to use the last layer of the UNet for classification.
            condition_discriminator (bool): Whether to condition the discriminator on the text prompt.
            device (torch.device, optional): The device to run the model on. Defaults to None, which uses the current device.
        """
        super().__init__()
        if gan_model_arch == 'sd_1_5':
            pretrain_model = "runwayml/stable-diffusion-v1-5"
        elif gan_model_arch == 'sd_2_1':
            pretrain_model = "stabilityai/stable-diffusion-2-1"
        else:
            raise ValueError(f"Unsupported discriminator architecture: {gan_model_arch}")
        self.D_args = {
            "model_name": gan_model_arch,
            "scheduler": scheduler,
            "weight_dtype": weight_dtype,
            "is_D": True,
            "full_finetuning": full_finetuning,
            "train_text_encoder_lora": False,
            "tune_text_encoder": False,
            "tune_vae": False,
            "pretrain_model": pretrain_model,
        }
        self.condition_discriminator = condition_discriminator
        try:
            self.D_sd_pipeline = load_pipeline(**self.D_args)
        except Exception as e:
            if gan_model_arch == 'sd_2_1':
                self.D_args['pretrain_model'] = "hanker917/stable-diffusion-2-1"
            self.D_sd_pipeline = load_pipeline(**self.D_args)
        # print("D_sd pipeline", self.D_sd_pipeline)

        self.weight_dtype = weight_dtype

        self.unet = self.D_sd_pipeline.unet
        
        self.ori_scheduler = copy.deepcopy(self.D_sd_pipeline.scheduler)
        
        # for classification
        self.gan_unet_lastlayer_cls = gan_unet_lastlayer_cls
        if self.gan_unet_lastlayer_cls:
            ori_last_conv = self.D_sd_pipeline.unet.conv_out
            self.mlp = nn.Conv2d(ori_last_conv.in_channels, 1, ori_last_conv.kernel_size, ori_last_conv.padding, ori_last_conv.stride, dtype=torch.float32)
            self.unet.conv_out = self.mlp # to hack in accelerator.prepare
        else: # MLP
            self.mlp = nn.Sequential(
                nn.Linear(4, 1, dtype=torch.float32)
            )
        self.cls_loss_fn = nn.BCEWithLogitsLoss()
        self.D_sd_parameters =  _get_trainable_parameters(
            self.D_sd_pipeline, 
            is_D=True, 
            full_finetuning=self.D_args['full_finetuning']
            )
        self.mlp_parameters = [p for p in self.mlp.parameters()]

    
    def get_trainable_parameters(self):
        """
        Get the trainable parameters of the discriminator. All trainable parameters
        include the UNet(LoRA or full) and the MLP for classification.
        """
        return self.D_sd_parameters + self.mlp_parameters

    def set_D_sd_pipeline_lora(self, requires_grad=True):
        """
        Set the training parameters in the discriminator's SD pipeline.

        Args:
            requires_grad (bool): Whether to train the parameters.
        """
        for p in self.D_sd_parameters:
            p.requires_grad = requires_grad

    def get_D_gt_noise(self, device, **kwargs):
        """
        Get the ground truth noise for the discriminator.

        Args:
            device (torch.device): The device to run the model on.
            batch (dict): A dictionary containing the batch data, including 'latents'.
        """
        ori_latents = kwargs['batch']['latents'].to(device, dtype=self.weight_dtype)
        return ori_latents

    def D_sd_pipeline_forward(self, training_latents, side='G',**kwargs):
        """
        Forward pass for the discriminator's SD pipeline.

        Args:
            training_latents (torch.Tensor): The input latents for the discriminator.
            side (str): The side of the GAN ('G' for generator, 'D' for discriminator).
            prompt (str): The text prompt for conditioning the discriminator.
            num_inference_steps (int): The number of inference steps for the scheduler.
            negative_prompt_embeds (torch.Tensor): The negative prompt embeddings for conditioning.

        Returns:
            gan_loss (torch.Tensor): The GAN loss computed by the discriminator.
        """
        device = training_latents.device
        if side == 'G':
            
            # set D_sd_pipeline no grad
            self.unet.eval()
            self.set_D_sd_pipeline_lora(requires_grad=False)

            # get discriminator condition
            if self.condition_discriminator:
                D_cond = self.D_sd_pipeline.encode_prompt(
                    prompt=kwargs['prompt'], 
                    device=device, 
                    num_images_per_prompt=1, 
                    do_classifier_free_guidance=False)[0]
            else:
                D_cond = kwargs['negative_prompt_embeds']

            self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
            timesteps = self.ori_scheduler.timesteps
            training_latents = self.ori_scheduler.scale_model_input(training_latents, timesteps[-1]) # identity in ddpm

            noise_pred = self.unet(
                training_latents,
                timesteps[-1], # marking its own step, could be out of domain
                encoder_hidden_states=D_cond,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0]
            
            noise_pred = noise_pred.permute(0,2,3,1) # -> (bs, h, w, 4)
            if self.gan_unet_lastlayer_cls:
                pred = noise_pred # noise_pred -> (bs, h, w, 1)
            else:
                pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
            target = torch.ones_like(pred) # -> (bs, h, w, 2)

            with torch.autocast('cuda'):
                gan_loss = self.cls_loss_fn(pred, target)
            return gan_loss

            
        elif side == 'D':

            self.unet.train()
            self.set_D_sd_pipeline_lora(requires_grad=True)
            training_latents.requires_grad_(False)

            D_cond = torch.cat(
                [kwargs['negative_prompt_embeds'], kwargs['negative_prompt_embeds']]
            )

            with torch.no_grad():
                ori_latents = self.get_D_gt_noise(device, **kwargs)
            

            self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
            timesteps = self.ori_scheduler.timesteps
            
            input_latents = torch.cat([training_latents, ori_latents])
            input_latents = self.ori_scheduler.scale_model_input(input_latents, timesteps[-1]) # identity in ddpm

            noise_pred = self.unet(
                input_latents,
                timesteps[-1], # marking its own step, could be out of domain
                encoder_hidden_states=D_cond,
                cross_attention_kwargs=None,
                return_dict=False,
            )[0] # (2*bs, 4, h, w)

                
            noise_pred = noise_pred.permute(0,2,3,1) # -> (2*bs, h, w, 4)
            if self.gan_unet_lastlayer_cls:
                pred = noise_pred # noise_pred -> (bs, h, w, 1)
            else:
                pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
            target = torch.ones_like(pred) # -> (2*bs, h, w, 1)

            target[:target.shape[0]//2] = 0 # label = 0 for generated_image

            with torch.autocast('cuda'):
                gan_loss = self.cls_loss_fn(pred, target)
            return gan_loss

    @torch.no_grad()
    def encode_prompt(self, prompt, device, batch_size, do_classifier_free_guidance=False):
        if isinstance(self.D_sd_pipeline, TrainableSDPipeline):
            null_embed = self.D_sd_pipeline.encode_prompt(
                prompt, 
                device, 
                batch_size, 
                do_classifier_free_guidance=do_classifier_free_guidance
            )[0]
            pooled_null_embed = None
        # TODO: support TrainableSDXLPipeline
        # elif isinstance(self.D_sd_pipeline, TrainableSDXLPipeline):
        #     null_embed, _, pooled_null_embed, _ = self.D_sd_pipeline.encode_prompt(
        #         prompt, 
        #         device=device, 
        #         num_images_per_prompt=batch_size, 
        #         do_classifier_free_guidance=do_classifier_free_guidance
        #     )
        # assume this function will be only called once
        # self.D_sd_pipeline.text_encoder.to('cpu')
        # self.D_sd_pipeline.vae.to('cpu')

        return null_embed, pooled_null_embed

    def to(self, *args, **kwargs):
        self.D_sd_pipeline.to(*args, **kwargs)
        return super().to(*args, **kwargs)


# class D_sdxl(D_sd):
#     def __init__(self, args, weight_dtype, device=None) -> None:
#         super().__init__()
#         self.D_args = copy.deepcopy(args)
#         self.D_args.train_text_encoder_lora = False
#         self.D_args.tune_text_encoder = False
#         self.D_args.pretrain_model = 'stabilityai/stable-diffusion-xl-base-1.0'
#         self.D_sd_pipeline = load_pipeline(self.D_args, args.gan_model_arch, weight_dtype, is_D=True).to(device)

#         self.weight_dtype = weight_dtype

#         self.unet = self.D_sd_pipeline.unet

#         if args.train_text_encoder_lora or args.tune_text_encoder:
#             self.D_sd_pipeline.text_encoder.to(device)
#             self.text_encoder = self.D_sd_pipeline.text_encoder
        
#         self.ori_scheduler = copy.deepcopy(self.D_sd_pipeline.scheduler)
        
#         # for classification
#         if args.gan_unet_lastlayer_cls:
#             ori_last_conv = self.D_sd_pipeline.unet.conv_out
#             self.mlp = nn.Conv2d(ori_last_conv.in_channels, 1, ori_last_conv.kernel_size, ori_last_conv.padding, ori_last_conv.stride)
#             self.unet.conv_out = self.mlp # to hack in accelerator.prepare
#         else: # MLP
#             self.mlp = nn.Sequential(
#                 nn.Linear(4, 1)
#             )
#         self.cls_loss_fn = nn.BCEWithLogitsLoss()

#         if args.train_text_encoder_lora or args.tune_text_encoder:
#             self.D_sd_pipeline.text_encoder_2.to(device)
#             self.text_encoder_2 = self.D_sd_pipeline.text_encoder_2

#         # used in add_time_ids
#         height = args.resolution or self.D_sd_pipeline.default_sample_size * self.D_sd_pipeline.vae_scale_factor
#         width = args.resolution or self.D_sd_pipeline.default_sample_size * self.D_sd_pipeline.vae_scale_factor

#         self.original_size = (height, width)
#         self.target_size = (height, width)
#         self.crops_coords_top_left = (0, 0)
#         self.add_time_ids = self._get_add_time_ids(
#             self.original_size,
#             self.crops_coords_top_left,
#             self.target_size,
#             weight_dtype
#         ).to(device)


#     def D_sd_pipeline_forward(self, training_latents, side='G',**kwargs):
#         device = training_latents.device
#         bs = training_latents.shape[0]

#         if side == 'G':
            
#             # set D_sd_pipeline no grad
#             self.unet.eval()
#             self.set_D_sd_pipeline_lora(requires_grad=False)

#             D_cond = kwargs['negative_prompt_embeds']

#             add_time_ids = self.add_time_ids.repeat(bs, 1)
#             # Predict the noise residual
#             unet_added_conditions = {"time_ids": add_time_ids}
#             unet_added_conditions.update({"text_embeds": kwargs['negative_pooled_prompt_embeds']})

#             self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
#             timesteps = self.ori_scheduler.timesteps
#             training_latents = self.ori_scheduler.scale_model_input(training_latents, timesteps[-1]) # identity in ddpm

#             noise_pred = self.unet(
#                 training_latents,
#                 timesteps[-1], # marking its own step, could be out of domain
#                 encoder_hidden_states=D_cond,
#                 added_cond_kwargs=unet_added_conditions,
#                 return_dict=False,
#             )[0]
            
#             # noise_pred = noise_pred.to(kwargs['negative_prompt_embeds'].dtype).flatten(-2).permute(0,2,1) # -> (2*bs,h*w,4)
#             noise_pred = noise_pred.permute(0,2,3,1) # -> (bs, h, w, 4)
#             if self.D_args.gan_unet_lastlayer_cls:
#                 pred = noise_pred # noise_pred -> (bs, h, w, 1)
#             else:
#                 pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
#             target = torch.ones_like(pred) # -> (bs, h, w, 2)

#             with torch.autocast('cuda'):
#                 gan_loss = self.cls_loss_fn(pred, target)
#             return gan_loss

            
#         elif side == 'D':

#             self.unet.train()
#             self.set_D_sd_pipeline_lora(requires_grad=True)
#             training_latents.requires_grad_(False)

#             D_cond = torch.cat(
#                 [kwargs['negative_prompt_embeds'], kwargs['negative_prompt_embeds']]
#             )

#             with torch.no_grad():
#                 ori_latents = self.get_D_gt_noise(device, **kwargs)

#             add_time_ids = self.add_time_ids.repeat(2*bs, 1)
#             # Predict the noise residual
#             unet_added_conditions = {"time_ids": add_time_ids}
#             unet_added_conditions.update(
#                 {"text_embeds": torch.cat([
#                     kwargs['negative_pooled_prompt_embeds'], kwargs['negative_pooled_prompt_embeds']
#                 ], dim=0)}
#             )

#             self.ori_scheduler.set_timesteps(kwargs['num_inference_steps'], device=device)
#             timesteps = self.ori_scheduler.timesteps

#             input_latents = torch.cat([training_latents, ori_latents])
#             input_latents = self.ori_scheduler.scale_model_input(input_latents, timesteps[-1]) # identity in ddpm

#             noise_pred = self.unet(
#                 input_latents,
#                 timesteps[-1], # marking its own step, could be out of domain
#                 encoder_hidden_states=D_cond,
#                 added_cond_kwargs=unet_added_conditions,
#                 return_dict=False,
#             )[0] # (2*bs, 4, h, w)

#             noise_pred = noise_pred.permute(0,2,3,1) # -> (2*bs, h, w, 4)
#             if self.D_args.gan_unet_lastlayer_cls:
#                 pred = noise_pred # noise_pred -> (bs, h, w, 1)
#             else:
#                 pred = self.mlp(noise_pred) # -> (bs, h, w, 2)
#             target = torch.ones_like(pred) # -> (2*bs, h, w, 1)
#             target[:target.shape[0]//2] = 0 # label = 0 for generated_image

#             with torch.autocast('cuda'):
#                 gan_loss = self.cls_loss_fn(pred, target)
#             return gan_loss

#     @torch.no_grad()
#     def encode_prompt(self, prompt, device, batch_size, do_classifier_free_guidance=False):
#         null_embed, pooled_null_embed = super().encode_prompt(prompt, device, batch_size, do_classifier_free_guidance)
#         self.D_sd_pipeline.text_encoder_2.to('cpu')

#         return null_embed, pooled_null_embed

#     def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
#         from torch.nn.parallel import DistributedDataParallel
#         add_time_ids = list(original_size + crops_coords_top_left + target_size)

#         if isinstance(self.unet, DistributedDataParallel):
#             passed_add_embed_dim = (
#                 self.unet.module.config.addition_time_embed_dim * len(add_time_ids) + self.D_sd_pipeline.text_encoder_2.config.projection_dim
#             )
#             expected_add_embed_dim = self.unet.module.add_embedding.linear_1.in_features
#         else:
#             passed_add_embed_dim = (
#                 self.unet.config.addition_time_embed_dim * len(add_time_ids) + self.D_sd_pipeline.text_encoder_2.config.projection_dim
#             )
#             expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

#         if expected_add_embed_dim != passed_add_embed_dim:
#             raise ValueError(
#                 f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
#             )

#         add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
#         return add_time_ids
