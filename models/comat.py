import torch
import random
from lightning.pytorch import LightningModule

from diffusers.optimization import get_scheduler
from diffusers.callbacks import PipelineCallback, MultiPipelineCallbacks
from diffusers.image_processor import PipelineImageInput

from typing import List, Dict, Any, Union, Optional, Callable

from .components.pipeline.load_pipeline import (
    load_pipeline,
    get_trainable_parameters,
    TrainableSDPipeline,
    AttrConcenTrainableSDPipeline,
)

class CoMat(LightningModule):
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
            caption_model_names: List[str],
            weights: List[float],
            gan_loss: bool,
            gan_model_arch: str,
            gan_loss_weight: float,
            gan_unet_lastlayer_cls: bool,
            condition_discriminator: bool,
            resolution: int,
            K: int,
            total_step: int,
            do_classifier_free_guidance: bool,
            train_batch_size: int,
            mask_token_loss_weight: float,
            mask_pixel_loss_weight: float,
            norm_grad: bool,
            attrcon_train_steps: int,
            cfg_scale: float = 7.5,
            cfg_rescale: float = 0.0,
            # Training parameters
            learning_rate: float = 1e-5,
            learning_rate_D: float = 1e-5,
            gradient_accumulation_steps: int = 4,
            lr_scheduler: str = "constant",
            lr_warmup_steps: int = 0,
            adam_beta1: float = 0.9,
            adam_beta2: float = 0.999,
            adam_weight_decay: float = 1e-2,
            adam_epsilon: float = 1e-08,
            max_grad_norm: float = 1.0,
            adam_beta1_D: float = 0.9,
            adam_beta2_D: float = 0.999,
            max_grad_norm_D: float = 1.0,
            inference_only: bool = False,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.inference_only = inference_only
        if self.hparams.weight_dtype == 'fp16':
            self.hparams.weight_dtype = torch.float16
        elif self.hparams.weight_dtype == 'fp32':
            self.hparams.weight_dtype = torch.float32
        elif self.hparams.weight_dtype == 'bf16':
            self.hparams.weight_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported weight_dtype: {self.hparams.weight_dtype}")
        # pipeline
        self.pipeline = load_pipeline(
            pretrain_model=self.hparams.pretrain_model,
            model_name=self.hparams.model_name,
            weight_dtype=self.hparams.weight_dtype,
            full_finetuning=self.hparams.full_finetuning,
            tune_vae=self.hparams.tune_vae,
            tune_text_encoder=self.hparams.tune_text_encoder,
            train_text_encoder_lora=self.hparams.train_text_encoder_lora,
            lora_rank=self.hparams.lora_rank,
            is_D=False,
            revision=self.hparams.revision,
            scheduler=self.hparams.scheduler,
            gradient_checkpointing=self.hparams.gradient_checkpointing,
        )
        # self.pipeline.enable_xformers_memory_efficient_attention()
        if not self.inference_only:
            self._init_training_components()

        # 初始化null embeddings
        self.register_buffer("null_embed", None)
        self.register_buffer("pooled_null_embed", None)
        self.register_buffer("gan_null_embed", None)
        self.register_buffer("gan_pooled_null_embed", None)

        # step counter for caption model
        self.step_count = 0

        # trainable keys
        self.trainable_keys = self.get_trainable_keys()

        # set automatic optimization to False
        self.automatic_optimization = False

        # Register the backbone components
        backbone = {
            "unet": self.pipeline.unet,
            "vae": self.pipeline.vae,
            "text_encoder": self.pipeline.text_encoder,
        }
        if getattr(self.pipeline, "safety_checker", None) is not None:
            backbone["safety_checker"] = self.pipeline.safety_checker
        self.backbone = torch.nn.ModuleDict(backbone)

    def _init_training_components(self):
        from .components.caption_model.CaptionModel import CaptionModelWrapper
        from .components.discriminator.load_discriminator import load_discriminator
        from .components.seg_model.gsam import GsamSegModel
        from .utils.attn.tc_attn_utils import AttentionStore, register_attention_control

        self.caption_model = CaptionModelWrapper(
            caption_model_names=self.hparams.caption_model_names,
            weights=self.hparams.weights,
            dtype=self.hparams.weight_dtype
        )

        if self.hparams.gan_loss:
            self.D = load_discriminator(
                gan_model_arch=self.hparams.gan_model_arch,
                weight_dtype=self.hparams.weight_dtype,
                scheduler=self.hparams.scheduler,
                full_finetuning=self.hparams.full_finetuning,
                gan_unet_lastlayer_cls=self.hparams.gan_unet_lastlayer_cls,
                condition_discriminator=self.hparams.condition_discriminator
            )

        if 'attrcon' in self.hparams.model_name:
            if 'sdxl' in self.hparams.model_name:
                train_layer_ls = ['mid_16', 'up_16', 'up_32']
            else:
                train_layer_ls = ['mid_8', 'up_16', 'up_32', 'up_64']
            self.seg_model = GsamSegModel(train_layer_ls=train_layer_ls)
            self.pipeline.controller = AttentionStore(train_layer_ls)
            register_attention_control(self.pipeline.unet, self.pipeline.controller)

    def setup(self, stage: str):
        """Setup method called at the beginning of fit/validate/test/predict"""
        if self.inference_only:
            return
        if stage == "fit":
            # Initialize null embeddings
            if self.hparams.do_classifier_free_guidance:
                with torch.no_grad():
                    if isinstance(self.pipeline, TrainableSDPipeline):
                        self.null_embed = self.pipeline.encode_prompt("", self.pipeline.device, self.hparams.train_batch_size, do_classifier_free_guidance=False)[0]
                    # elif isinstance(self.pipeline, TrainableSDXLPipeline):
                    #     self.null_embed, _, self.pooled_null_embed, _ = self.pipeline.encode_prompt("", device=self.device, num_images_per_prompt=self.hparams.train_batch_size, do_classifier_free_guidance=False)
                    
                    if self.hparams.gan_loss:
                        self.gan_null_embed, self.gan_pooled_null_embed = self.D.encode_prompt("", self.pipeline.device, self.hparams.train_batch_size, do_classifier_free_guidance=False)
                        try:
                            del self.D.D_sd_pipeline.config['vae']  # Remove VAE from discriminator config
                            del self.D.D_sd_pipeline.config['text_encoder']  # Remove text encoder from discriminator config
                        except Exception as e:
                            del self.D.D_sd_pipeline.vae
                            del self.D.D_sd_pipeline.text_encoder

            # # Clean up unnecessary components
            # if self.hparams.gan_loss:
            #     if 'vae' in self.D.D_sd_pipeline.config:
            #         del self.D.D_sd_pipeline.config['vae']
            #     # if hasattr(self.D.D_sd_pipeline, 'text_encoder'):
            #     #     del self.D.D_sd_pipeline.text_encoder
            #     # if isinstance(self.D.D_sd_pipeline, TrainableSDXLPipeline):
            #     #     del self.D.D_sd_pipeline.text_encoder_2
            #     torch.cuda.empty_cache()
    
    def configure_optimizers(self):
        """Configure optimizers for Lightning"""
        if self.inference_only:
            raise RuntimeError("CoMat was initialized with inference_only=True and cannot be trained.")
        # Get trainable parameters
        if self.hparams.train_text_encoder_lora:
            G_parameters, text_lora_parameters = get_trainable_parameters(
                self.pipeline, 
                is_D=False,
                full_finetuning=self.hparams.full_finetuning,
                tune_vae=self.hparams.tune_vae,
                tune_text_encoder=self.hparams.tune_text_encoder,
                train_text_encoder_lora=self.hparams.train_text_encoder_lora, 
                )
            G_parameters.extend(text_lora_parameters)
        else:
            G_parameters = get_trainable_parameters(
                self.pipeline, 
                is_D=False,
                full_finetuning=self.hparams.full_finetuning,
                tune_vae=self.hparams.tune_vae,
                tune_text_encoder=self.hparams.tune_text_encoder,
                train_text_encoder_lora=self.hparams.train_text_encoder_lora, 
                )

        # Generator optimizer
        optimizer_G = torch.optim.AdamW(
            G_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),
            weight_decay=self.hparams.adam_weight_decay,
            eps=self.hparams.adam_epsilon,
        )
        # Learning rate scheduler
        lr_scheduler = get_scheduler(
            name=self.hparams.lr_scheduler,
            optimizer=optimizer_G,
            num_warmup_steps=self.hparams.lr_warmup_steps * self.hparams.gradient_accumulation_steps,
        )

        if self.hparams.gan_loss:
            # Discriminator optimizer
            D_parameters = self.D.get_trainable_parameters()
            optimizer_D = torch.optim.AdamW(
                D_parameters,
                lr=self.hparams.learning_rate_D,
                betas=(self.hparams.adam_beta1_D, self.hparams.adam_beta2_D),
                weight_decay=self.hparams.adam_weight_decay,
                eps=self.hparams.adam_epsilon,
            )
            return (
                {
                    "optimizer": optimizer_G,
                    "lr_scheduler": {
                        "scheduler": lr_scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                },
                {
                    "optimizer": optimizer_D,
                }
            )
        else:
            return optimizer_G

    def on_train_epoch_start(self):
        """Called at the start of training epoch"""
        if self.inference_only:
            return
        self.pipeline.unet.train()
        if self.hparams.tune_text_encoder or self.hparams.train_text_encoder_lora:
            self.pipeline.text_encoder.train()
            if hasattr(self.pipeline, "text_encoder_2"):
                self.pipeline.text_encoder_2.train()

    def training_step(self, batch, batch_idx):
        """Training step for Lightning"""
        if self.inference_only:
            raise RuntimeError("CoMat was initialized with inference_only=True and cannot be trained.")
        from .utils.attribute_concen_utils import get_attention_map_index_to_wordpiece

        if self.hparams.gan_loss:
            optimizer_g, optimizer_d = self.optimizers()
        else:
            optimizer_g = self.optimizers()
        sch = self.lr_schedulers()

        # Generator step
        self.toggle_optimizer(optimizer_g)
        # Generate training timesteps
        interval = self.hparams.total_step // self.hparams.K
        max_start = self.hparams.total_step - interval * (self.hparams.K - 1) - 1
        start = random.randint(0, max_start)
        training_steps = list(range(start, self.hparams.total_step, interval))
        
        # Get null embeddings if training text encoder
        null_embed = self.null_embed
        pooled_null_embed = self.pooled_null_embed
        if self.hparams.tune_text_encoder or self.hparams.train_text_encoder_lora:
            if isinstance(self.pipeline, TrainableSDPipeline):
                null_embed = self.pipeline.encode_prompt("", self.pipeline.device, self.hparams.train_batch_size, do_classifier_free_guidance=False)[0]
            # elif isinstance(self.pipeline, TrainableSDXLPipeline):
            #     null_embed, _, pooled_null_embed, _ = self.pipeline.encode_prompt("", device=self.device, num_images_per_prompt=self.hparams.train_batch_size, do_classifier_free_guidance=False)

        # Pipeline forward kwargs
        kwargs = dict(
            prompt=batch["text"],
            height=self.hparams.resolution,
            width=self.hparams.resolution,
            training_timesteps=training_steps,
            detach_gradient=True,
            train_text_encoder=self.hparams.tune_text_encoder or self.hparams.train_text_encoder_lora,
            num_inference_steps=self.hparams.total_step,
            guidance_scale=self.hparams.cfg_scale,
            guidance_rescale=self.hparams.cfg_rescale,
            negative_prompt_embeds=null_embed if self.hparams.do_classifier_free_guidance else None,
            early_exit=False,
            return_latents=True if self.hparams.gan_loss else False,
        )
        
        # Add attrcon steps if needed
        if 'attrcon' in self.hparams.pretrain_model:
            kwargs['attrcon_train_steps'] = random.choices(training_steps, k=min(self.hparams.attrcon_train_steps, len(training_steps)))

        # Pipeline forward
        if isinstance(self.pipeline, TrainableSDPipeline):
            if self.hparams.gan_loss:
                image, training_latents = self.pipeline.forward(bp_on_trained=True, double_laststep=False, fast_training=False, **kwargs)
            else:
                image = self.pipeline.forward(bp_on_trained=True, double_laststep=False, fast_training=False, **kwargs)
        # elif isinstance(self.pipeline, TrainableSDXLPipeline):
        #     if self.hparams.gan_loss:
        #         image, training_latents = self.pipeline.forward(negative_pooled_prompt_embeds=pooled_null_embed if self.hparams.do_classifier_free_guidance else None, **kwargs)
        #     else:
        #         image = self.pipeline.forward(negative_pooled_prompt_embeds=pooled_null_embed if self.hparams.do_classifier_free_guidance else None, **kwargs)
        else:
            raise NotImplementedError("This model is not supported yet")

        # Calculate reward loss
        offset_range = self.hparams.resolution // 224
        random_offset_x = random.randint(0, offset_range)
        random_offset_y = random.randint(0, offset_range)
        size = self.hparams.resolution - offset_range
        
        caption_rewards = self.caption_model(
            image[:,:,random_offset_x:random_offset_x + size, random_offset_y:random_offset_y + size].to(self.hparams.weight_dtype), 
            prompts=batch['text'],
            step=self.step_count,
            text_encoder=self.pipeline.text_encoder,
            batch=batch
        )
        self.step_count += 1

        g_loss = -caption_rewards["combined"].mean()

        # Add GAN loss
        if self.hparams.gan_loss:
            kwargs['negative_prompt_embeds'] = self.gan_null_embed
            kwargs['negative_pooled_prompt_embeds'] = self.gan_pooled_null_embed
            
            G_loss = self.D.D_sd_pipeline_forward(training_latents, side='G', **kwargs)
            g_loss += self.hparams.gan_loss_weight * G_loss
            self.log('G_loss', G_loss, prog_bar=True)

        # Add attribution concentration loss
        if 'attrcon' in self.hparams.model_name:
            all_subtree_indices = [self.pipeline._extract_attribution_indices(p) for p in batch['text']]
            attn_map_idx_to_wp_all = [get_attention_map_index_to_wordpiece(self.pipeline.tokenizer, p) for p in batch['text']]
            attn_map = self.pipeline.attn_dict

            token_loss, pixel_loss, grounding_loss_dict = self.seg_model.get_mask_loss(
                image.clamp(0, 1),
                batch['text'],
                all_subtree_indices,
                attn_map_idx_to_wp_all,
                attn_map
            )
            g_loss += self.hparams.mask_token_loss_weight * token_loss
            g_loss += self.hparams.mask_pixel_loss_weight * pixel_loss
            
            self.pipeline.attn_dict = {}  # clear the attn_dict after usage
            self.log('token_loss', token_loss, prog_bar=True)
            self.log('pixel_loss', pixel_loss, prog_bar=True)

        # Register gradient hook for monitoring
        norm = {}
        def record_grad(grad):
            norm['reward_norm'] = grad.norm(2).item()
            if self.hparams.norm_grad:
                grad = grad / (norm['reward_norm'] / 1e4)  # 1e4 for numerical stability
            return grad
        
        image.register_hook(record_grad)

        # Log metrics
        self.log('g_loss', g_loss, prog_bar=True)
        for k, v in caption_rewards.items():
            self.log(f'reward_{k}', v.mean(), prog_bar=True)
        
        # Log gradient norm after backward pass
        g_loss_scaled = g_loss / self.hparams.gradient_accumulation_steps
        self.log_dict(norm, prog_bar=True)
        self.manual_backward(g_loss_scaled)
        self.clip_gradients(optimizer_g, gradient_clip_val=self.hparams.max_grad_norm)
        if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0:
            optimizer_g.step()
            optimizer_g.zero_grad()
            # Step the learning rate scheduler
            sch.step()
        self.untoggle_optimizer(optimizer_g)

        # Discriminator step
        self.toggle_optimizer(optimizer_d)
        kwargs['batch'] = batch
        d_loss = self.D.D_sd_pipeline_forward(training_latents.detach(), side='D', **kwargs)
        self.log('d_loss', d_loss, prog_bar=True)
        d_loss_scaled = d_loss / self.hparams.gradient_accumulation_steps
        self.manual_backward(d_loss_scaled)
        self.clip_gradients(optimizer_d, gradient_clip_val=self.hparams.max_grad_norm_D)
        if (batch_idx + 1) % self.hparams.gradient_accumulation_steps == 0:
            optimizer_d.step()
            optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)


    # def validation_step(self, batch, batch_idx):
    #     """Validation step for Lightning"""
    #     self.pipeline.set_progress_bar_config(disable=True)
    #     images = self.pipeline(
    #         prompt=batch["text"],
    #         height=self.hparams.resolution,
    #         width=self.hparams.resolution,
    #         num_inference_steps=self.hparams.total_step,
    #         guidance_scale=self.hparams.cfg_scale,
    #         negative_prompt_embeds=self.null_embed if self.hparams.do_classifier_free_guidance else None,
    #         return_latents=False,  # We don't need latents in prediction
    #     ).images
    #     # if self.logger is not None:
    #     #     self.logger.experiment.add_images(
    #     #         "data/validation_images",
    #     #         images,
    #     #         self.global_step,
    #     #     )
    #     return images
        
    @torch.no_grad()
    def predict_step(self, batch, batch_idx):
        self.pipeline.unet.eval()
        if self.hparams.tune_text_encoder or self.hparams.train_text_encoder_lora:
            self.pipeline.text_encoder.eval()
            if hasattr(self.pipeline, "text_encoder_2"):
                self.pipeline.text_encoder_2.eval()

        # Generate images using the pipeline
        images = self.pipeline(
            prompt=batch["text"],
            height=self.hparams.resolution,
            width=self.hparams.resolution,
            num_inference_steps=self.hparams.total_step,
            guidance_scale=self.hparams.cfg_scale,
            negative_prompt_embeds=self.null_embed if self.hparams.do_classifier_free_guidance else None,
            return_latents=False,  # We don't need latents in prediction
        )
        # Convert images to numpy or PIL format if needed
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        elif isinstance(images, list):
            images = [img.cpu().numpy() for img in images]
        return images
    
    @torch.no_grad()
    def predict(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
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
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
        )


    def to(self, *args, **kwargs):
        self.pipeline.to(*args, **kwargs)
        if hasattr(self, 'D'):
            self.D.to(*args, **kwargs)
        return super().to(*args, **kwargs)
    
    def get_trainable_keys(self):
        trainable_keys = set()
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_keys.add(name)
        return trainable_keys

    def state_dict(self):
        # Only keep trainable parameters in the state dict
        state = super().state_dict()
        return {k: v for k, v in state.items() if k in self.trainable_keys}
    
    def pil_to_numpy(self, pil_image):
        return self.pipeline.image_processor.pil_to_numpy(pil_image)

    def numpy_to_pt(self, numpy_array):
        return self.pipeline.image_processor.numpy_to_pt(numpy_array)
    
    def load_checkpoint(self, checkpoint_path, need_train=False):
        """Load a checkpoint into the model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Checkpoint loaded from {checkpoint_path}")
        if not need_train:
            if hasattr(self, 'D'):
                del self.D
            if hasattr(self, 'caption_model'):
                del self.caption_model
            if hasattr(self, 'seg_model'):
                del self.seg_model
            torch.cuda.empty_cache()

    
