from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
from diffusers.loaders import LoraLoaderMixin,TextualInversionLoaderMixin, SD3LoraLoaderMixin
from typing import * 
import torch 
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from  diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3PipelineOutput,
    PipelineCallback, MultiPipelineCallbacks,
    PipelineImageInput,
    calculate_shift,
    retrieve_timesteps,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg


import pdb

class SpecialToken:
    text = ""
    index_in_all_tokens = 0
    steps = []
    weight = 1.0
    def __init__(self,text,index_in_all_tokens,steps,weight) -> None:
        self.text = text
        self.index_in_all_tokens = index_in_all_tokens
        self.steps = steps
        self.weight = weight

    def __repr__(self) -> str:
        return f"Text:{self.text}\tIndex of all:{self.index_in_all_tokens}\tSteps:{self.steps}\tWeight:{self.weight}\n"

def delete_elements(list_input, indexes):
    indexes.sort(reverse=True)
    index_mapping = {}
    for i in range(len(list_input)):
        if i in indexes:
            index_mapping[i] = None
        else:
            index_mapping[i] = i - len([j for j in indexes if j < i])

    for index in indexes:
        if index < len(list_input):
            del list_input[index]
    return list_input, index_mapping

class DynamicPromptPipeline():
    def parse_single_prompt(self, prompt:str, num_inference_steps:int):
        prompt = prompt.replace("[ "," [")

        # parse single prompt with special format. return clean prompt,special token indexes and their time steps and weights
        current = ""
        special_tokens = []
        ind = 0
        index_in_all_tokens = 0
        while ind < len(prompt):
            if prompt[ind]!="[":
                current += prompt[ind]
                ind += 1

            else:
                # enter special token
                # step1: tokenize current
                current_tokens = self.tokenizer.tokenize(current)
                

                index_in_all_tokens = len(current_tokens)
                special_token_str = ""
                ind +=1 
                while ind <len(prompt) and prompt[ind]!="]":
                    special_token_str += prompt[ind]
                    ind += 1
                if ind >=len(prompt) and prompt[ind-1]!="]":
                    ind += len(special_token_str)
                    current += special_token_str
                else:
                    res=special_token_str.split(":")
                    if len(res)==3:

                        text,steps_raw,weight_raw = res
                        try: 
                            endtimestep,starttimestep = map(lambda x:int(float(x)*num_inference_steps), steps_raw.split("-"))
                            
                            steps = list(range(endtimestep+1,starttimestep+1))[::-1]
                            steps = [num_inference_steps - i for i in steps]
                            weight = float(weight_raw)
                            text_tokens = self.tokenizer.tokenize(text)
                            for sub_ind, sub_token in enumerate(text_tokens):
                                sub_text = self.tokenizer.convert_tokens_to_string(sub_token)
                            
                                special_tokens.append(SpecialToken(sub_text, index_in_all_tokens + sub_ind,steps,weight))
                            index_in_all_tokens = index_in_all_tokens - 1 + len(text_tokens)
                            ind += 1
                            current += text 
                        except:
                            ind += len(special_token_str)
                            current += special_token_str
                    else:
                        ind += len(special_token_str)
                        current += special_token_str



                if len(self.tokenizer.tokenize(current)) <= index_in_all_tokens and special_tokens!=[]:
                    diff = index_in_all_tokens - len(self.tokenizer.tokenize(current)) + 2
                    merged_text = "" 
                    merged_weight = special_tokens[-1].weight 
                    merged_steps = special_tokens[-1].steps
                    merged_index_in_all_tokens = 0
                    for i in range(diff):
                        try:
                            spt = special_tokens.pop()
                            merged_text = spt.text + merged_text 
                            merged_index_in_all_tokens = spt.index_in_all_tokens
                        except:
                            break

                    special_tokens.append(SpecialToken(merged_text,merged_index_in_all_tokens,merged_steps,merged_weight)) 
        # print(special_tokens)
       
        clean_prompt_of_all = current 
        tokens_of_all = self.tokenizer.tokenize(clean_prompt_of_all)
 
        clean_prompt_and_specialtoken_weightindex_pair = {

        }
        clean_prompt_set = set()
        for step in range(num_inference_steps):
            tokens_of_step = [ t for t in tokens_of_all ]
            removed_indexes = []
            sp_list_of_step = []
            for sp_token in special_tokens:
                if step not in sp_token.steps:
                    removed_indexes.append(sp_token.index_in_all_tokens)
                else:
                    sp_list_of_step.append((sp_token.weight,sp_token.index_in_all_tokens))
            removed_indexes.sort(reverse=False)

            tokens_of_step, index_mapping = delete_elements(tokens_of_step, removed_indexes)

            sp_list_of_step_copy = sp_list_of_step.copy()
            # pdb.set_trace()
            for w,i in sp_list_of_step_copy:
                try:
                    sp_list_of_step.append((w,index_mapping[i]))
                except:
                    continue
            clean_prompt_of_step = self.tokenizer.convert_tokens_to_string(tokens_of_step)
            clean_prompt_and_specialtoken_weightindex_pair[step] = (clean_prompt_of_step, sp_list_of_step)
            clean_prompt_set.add(clean_prompt_of_step)
            
        return clean_prompt_and_specialtoken_weightindex_pair,clean_prompt_set


class StableDiffusionDynamicPromptPipeline(StableDiffusionPipeline, DynamicPromptPipeline):
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        num_inference_steps = 50
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        assert isinstance(prompt, str)
        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, LoraLoaderMixin):
            self._lora_scale = lora_scale

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

    
        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        ### Prompt parsing by Terry Zhang
        # step 1: find all special tokens that has [] surrounded, find their indexes after tokenizing
        if isinstance(prompt,str):
            prompts = [prompt]
        else:
            prompts = prompt 
        prompt_raw = prompt
        clean_prompt_and_specialtoken_weightindex_pair,clean_prompt_set = self.parse_single_prompt(prompt_raw,num_inference_steps) 
        
        clean_prompt_to_prompt_embeds = dict()
        for clean_prompt in clean_prompt_set:
            text_inputs = self.tokenizer(
                clean_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                # print(
                #     "The following part of your input was truncated because CLIP can only handle sequences up to"
                #     f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                # )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            clean_prompt_to_prompt_embeds[clean_prompt] = prompt_embeds
        


        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


        
        prompt_embeds_dict = {}
        
        for step in clean_prompt_and_specialtoken_weightindex_pair.keys():

            clean_prompt_of_step, sp_list_of_step = clean_prompt_and_specialtoken_weightindex_pair[step] 
            prompt_embeds_of_step = clean_prompt_to_prompt_embeds[clean_prompt_of_step]
            weight = torch.ones_like(prompt_embeds_of_step)

            for w,ind in sp_list_of_step:
                if ind!=None and ind+1<weight.size(1):
                    weight[:, ind+1] = w 
            previous_mean = prompt_embeds_of_step.float().mean(axis=[-2, -1]).to(prompt_embeds_of_step.dtype)
            prompt_embeds_dict[step] = prompt_embeds_of_step * weight
            current_mean = prompt_embeds_dict[step].float().mean(axis=[-2, -1]).to(prompt_embeds_dict[step].dtype)
            # prompt_embeds_dict[step]  *= (previous_mean / current_mean)
            expanded_previous_mean = previous_mean.unsqueeze(-1).unsqueeze(-1)  
            expanded_current_mean = current_mean.unsqueeze(-1).unsqueeze(-1)  

            
            expanded_previous_mean = expanded_previous_mean.expand_as(prompt_embeds_dict[step])
            expanded_current_mean = expanded_current_mean.expand_as(prompt_embeds_dict[step])
            
            prompt_embeds_dict[step] *= (expanded_previous_mean / expanded_current_mean)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                prompt_embeds_dict[step] = torch.cat([negative_prompt_embeds, prompt_embeds_dict[step]])
            # print(step,clean_prompt_of_step,weight[0,:,0])
        return prompt_embeds_dict

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        assert prompt_embeds is None 
        assert negative_prompt_embeds is None 
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        if isinstance(prompt,list):
            prompt_embeds_dict = None 
            for prompt_str in prompt:
                prompt_embeds_dict_one = self._encode_prompt(
                    prompt_str,
                    device,
                    num_images_per_prompt,
                    do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    lora_scale=text_encoder_lora_scale,
                    num_inference_steps=len(timesteps)
                )
                if prompt_embeds_dict is None:
                    prompt_embeds_dict = prompt_embeds_dict_one 
                else:
                    for k in prompt_embeds_dict.keys():
                        new_v = prompt_embeds_dict_one[k]
                        
                        if do_classifier_free_guidance:
                            neg_p, pos_p = torch.chunk(new_v,2)
                            neg_p_, pos_p_ = torch.chunk(prompt_embeds_dict[k],2)
                            neg_p_cat = torch.cat([neg_p_,neg_p])
                            pos_p_cat = torch.cat([pos_p_,pos_p])
                            prompt_embeds_dict[k] = torch.cat([neg_p_cat,pos_p_cat])
                        else:
                            prompt_embeds_dict[k] = torch.cat([prompt_embeds_dict[k],new_v])

        else:

            prompt_embeds_dict = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=text_encoder_lora_scale,
                num_inference_steps=len(timesteps)
            )

        # 4. Prepare timesteps


        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_dict[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        # with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds_dict[i],
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                # progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds_dict[0].dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
class StableDiffusion3DynamicPromptPipeline(StableDiffusion3Pipeline, DynamicPromptPipeline):
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        max_sequence_length: int = 256,
        lora_scale: Optional[float] = None,
        num_inference_steps: int = 50,
    ):
        assert isinstance(prompt, str)
        if lora_scale is not None and isinstance(self, SD3LoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        prompt_raw = prompt
        clean_prompt_and_specialtoken_weightindex_pair, clean_prompt_set = self.parse_single_prompt(
            prompt_raw, num_inference_steps
        )

        # compute clip and t5 embeddings for each unique clean prompt
        clean_prompt_to_prompt_embeds = {}
        clean_prompt_to_pooled = {}
        for clean_prompt in clean_prompt_set:
            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=clean_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=0,
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=clean_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                clip_model_index=1,
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=clean_prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
            clean_prompt_to_prompt_embeds[clean_prompt] = prompt_embeds
            clean_prompt_to_pooled[clean_prompt] = pooled_prompt_embeds

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt_2 = negative_prompt
            negative_prompt_3 =negative_prompt

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_2 = (
                batch_size * [negative_prompt_2] if isinstance(negative_prompt_2, str) else negative_prompt_2
            )
            negative_prompt_3 = (
                batch_size * [negative_prompt_3] if isinstance(negative_prompt_3, str) else negative_prompt_3
            )

            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                negative_prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=0,
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                negative_prompt_2,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                clip_model_index=1,
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=negative_prompt_3,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        if self.text_encoder is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, SD3LoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        prompts_embeds_dict = {}
        pooled_prompts_embeds_dict = {}
        for step in clean_prompt_and_specialtoken_weightindex_pair.keys():
            clean_prompt_of_step, sp_list_of_step = clean_prompt_and_specialtoken_weightindex_pair[step]
            prompt_embeds_of_step = clean_prompt_to_prompt_embeds[clean_prompt_of_step]
            pooled_prompt_embeds_of_step = clean_prompt_to_pooled[clean_prompt_of_step]

            weight = torch.ones_like(prompt_embeds_of_step)

            for w, ind in sp_list_of_step:
                if ind != None and ind + 1 < weight.size(1):
                    weight[:, ind + 1] = w
            previous_mean = prompt_embeds_of_step.float().mean(axis=[-2, -1]).to(prompt_embeds_of_step.dtype)
            prompt_embeds_of_step = prompt_embeds_of_step * weight
            current_mean = prompt_embeds_of_step.float().mean(axis=[-2, -1]).to(prompt_embeds_of_step.dtype)

            expanded_previous_mean = previous_mean.unsqueeze(-1).unsqueeze(-1)
            expanded_current_mean = current_mean.unsqueeze(-1).unsqueeze(-1)

            expanded_previous_mean = expanded_previous_mean.expand_as(prompt_embeds_of_step)
            expanded_current_mean = expanded_current_mean.expand_as(prompt_embeds_of_step)

            prompt_embeds_of_step *= (expanded_previous_mean / expanded_current_mean)

            prompts_embeds_dict[step] = prompt_embeds_of_step
            pooled_prompts_embeds_dict[step] = pooled_prompt_embeds_of_step

        return prompts_embeds_dict, negative_prompt_embeds, pooled_prompts_embeds_dict, negative_pooled_prompt_embeds


    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 256,
        skip_guidance_layers: List[int] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
        mu: Optional[float] = None,
    ):
        # same input checks and setup as before
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            prompt_3,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, height, width = latents.shape
            image_seq_len = (height // self.transformer.config.patch_size) * (
                width // self.transformer.config.patch_size
            )
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        
        # encode prompts
        if isinstance(prompt, list):
            prompt_embeds_dict = None 
            pooled_prompt_embeds_dict = None 
            negative_prompt_embeds_all = None
            negative_pooled_prompt_embeds_all = None
            for prompt_str in prompt:
                (
                    prompt_embeds_dict_one,
                    negative_prompt_embeds,
                    pooled_prompt_embeds_dict_one,
                    negative_pooled_prompt_embeds,
                ) = self._encode_prompt(
                    prompt_str,
                    device,
                    num_images_per_prompt,
                    self.do_classifier_free_guidance,
                    negative_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    clip_skip=clip_skip,
                    max_sequence_length=max_sequence_length,
                    lora_scale=lora_scale,
                    num_inference_steps=num_inference_steps,
                )
                if prompt_embeds_dict is None:
                    prompt_embeds_dict = prompt_embeds_dict_one 
                    pooled_prompt_embeds_dict = pooled_prompt_embeds_dict_one 
                    negative_prompt_embeds_all = negative_prompt_embeds
                    negative_pooled_prompt_embeds_all = negative_pooled_prompt_embeds
                else:
                    for k in prompt_embeds_dict.keys():
                        new_v = prompt_embeds_dict_one[k]
                        prompt_embeds_dict[k] = torch.cat([prompt_embeds_dict[k],new_v])
                    for k in pooled_prompt_embeds_dict.keys():
                        new_v = pooled_prompt_embeds_dict_one[k]
                        pooled_prompt_embeds_dict[k] = torch.cat([pooled_prompt_embeds_dict[k],new_v])
                    negative_prompt_embeds_all = torch.cat([negative_prompt_embeds_all,negative_prompt_embeds])
                    negative_pooled_prompt_embeds_all = torch.cat([negative_pooled_prompt_embeds_all,negative_pooled_prompt_embeds])
            negative_prompt_embeds = negative_prompt_embeds_all
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds_all

        else:
            (
                prompt_embeds_dict,
                negative_prompt_embeds,
                pooled_prompt_embeds_dict,
                negative_pooled_prompt_embeds,
            ) = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                clip_skip=clip_skip,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
                num_inference_steps=num_inference_steps,
            )

        if self.do_classifier_free_guidance:
            if skip_guidance_layers is not None:
                original_prompt_embeds_dict = prompt_embeds_dict
                original_pooled_prompt_embeds_dict = pooled_prompt_embeds_dict
            for step in prompt_embeds_dict.keys():
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                prompt_embeds_dict[step] = torch.cat([negative_prompt_embeds, prompt_embeds_dict[step]])
                pooled_prompt_embeds_dict[step] = torch.cat(
                    [negative_pooled_prompt_embeds, pooled_prompt_embeds_dict[step]]
                )

        

        # prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds_dict[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare image embeddings
        if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
            ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

            if self.joint_attention_kwargs is None:
                self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
            else:
                self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)

        # 7. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand latents for guidance when transformer expects doubled batch
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds_dict[i],
                    pooled_projections=pooled_prompt_embeds_dict[i],
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance (works whether embeddings were concatenated or not)
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    should_skip_layers = (
                        True
                        if i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                        else False
                    )
                    if skip_guidance_layers is not None and should_skip_layers:
                        timestep_inner = t.expand(latents.shape[0])
                        latent_model_input_inner = latents
                        noise_pred_skip_layers = self.transformer(
                            hidden_states=latent_model_input_inner,
                            timestep=timestep_inner,
                            encoder_hidden_states=original_prompt_embeds_dict[i],
                            pooled_projections=original_pooled_prompt_embeds_dict[i],
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = (
                            noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                        )

                # step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                # callbacks
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds_dict = callback_outputs.pop("prompt_embeds_dict", prompt_embeds_dict)
                    pooled_prompt_embeds_dict = callback_outputs.pop("pooled_prompt_embeds_dict", pooled_prompt_embeds_dict)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # decode
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # offload
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusion3PipelineOutput(images=image)






if __name__ == "__main__":
    from diffusers import UniPCMultistepScheduler
    # pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers")
    pipe = StableDiffusion3DynamicPromptPipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    pipe.to("cuda")


    prompt_list=[
        # "a red horse on the yellow grass, anime style",
    # "a red horse on the yellow grass, [anime:0-1:1.5] style",
    # "a red horse on the yellow grass, detailed",
    "a cat [anime:0.2-0.8:1.5] on grass",
    "a red horse on the yellow grass, [detailed:0.85-1:1]"]
    pipe(prompt_list, num_inference_steps=20, guidance_scale=7.0, height=512, width=512)

    # for i in range(len(prompt_list)):
    #     prompt=prompt_list[i]
    #     image0 = pipe(prompt,generator=torch.Generator().manual_seed(1), num_inference_steps=10,).images[0]
    #     image0.save(f"{i:05}.png")