import os
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from typing import Union, List, Optional, Dict
from lightning.pytorch import LightningModule
from diffusers import StableDiffusionPipeline

from .comat import CoMat
from .components.refiner.refiner import RefinerModel
from .components.refiner.llmrefiner import (
    MistralRefiner, 
    MistralRefinerwithNLP, 
    MistralRefinerwithLM, 
    MistralRefinerwithMLLM,
    MistralRefinerwithClassname
)
from .components.sentinel.llmsentinel import MistralCleaner

class NoxEyePipeline(nn.Module):
    def __init__(self, 
                 sd_pretrain_model: str = "stabilityai/stable-diffusion-2-1",
                 sd_pretrain_model_config: Optional[Dict] = None,
                 sd_checkpoint_path: Optional[str] = None,
                 refine_pretrain_model: Optional[str] = None,
                 refine_pretrain_model_config: Optional[Dict] = None,
                 refine_checkpoint_path: Optional[str] = None,
                 sentinel_pretrain_model: Optional[str] = None,
                 sentinel_pretrain_model_config: Optional[Dict] = None,
                 sentinel_checkpoint_path: Optional[str] = None,
                 dtype: str = "fp16"):
        super().__init__()
        self.sd_pretrain_model = sd_pretrain_model
        self.sd_pretrain_model_config = sd_pretrain_model_config
        self.sd_checkpoint_path = sd_checkpoint_path
        self.refine_pretrain_model = refine_pretrain_model
        if dtype == "fp16":
            self.dtype = torch.float16
        elif dtype == "bf16":
            self.dtype = torch.bfloat16
        elif dtype == "fp32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        if sd_pretrain_model == 'comat':
            self.model = CoMat(**sd_pretrain_model_config).to(self.dtype)
            self.model.load_checkpoint(sd_checkpoint_path)
            self.image_processor = self.model.pipeline.image_processor
        else:
            self.model = StableDiffusionPipeline.from_pretrained(sd_pretrain_model, torch_dtype=self.dtype)
            self.image_processor = self.model.image_processor

        if refine_pretrain_model_config is None:
            refine_pretrain_model_config = {}
        if refine_pretrain_model is None:
            self.refiner = None
        elif 'Promptist' in refine_pretrain_model:
            print("Using Promptist as refiner")
            self.refiner = RefinerModel(pretrained_model_name=refine_pretrain_model, **refine_pretrain_model_config).to(self.dtype)
        elif 'mistral' in refine_pretrain_model and '+' not in refine_pretrain_model:
            print("Using Mistral as refiner")
            self.refiner = MistralRefiner(model_name=refine_pretrain_model, max_new_tokens=128, **refine_pretrain_model_config).to(self.dtype)
            if refine_checkpoint_path is not None:
                self.refiner.load_checkpoint(refine_checkpoint_path)
        elif 'mistral' in refine_pretrain_model and refine_pretrain_model.split('+')[-1] == 'nlp':
            print("Using Mistral with NLP as refiner")
            self.refiner = MistralRefinerwithNLP(model_name=refine_pretrain_model.split('+')[0], max_new_tokens=128, **refine_pretrain_model_config).to(self.dtype)
            if refine_checkpoint_path is not None:
                self.refiner.load_checkpoint(refine_checkpoint_path)
        elif 'mistral' in refine_pretrain_model and refine_pretrain_model.split('+')[-1] == 'lm':
            print("Using Mistral with LM as refiner")
            self.refiner = MistralRefinerwithLM(model_name=refine_pretrain_model.split('+')[0], max_new_tokens=128, **refine_pretrain_model_config).to(self.dtype)
            if refine_checkpoint_path is not None:
                self.refiner.load_checkpoint(refine_checkpoint_path)
        elif 'mistral' in refine_pretrain_model and refine_pretrain_model.split('+')[-1] == 'mllm':
            print("Using Mistral with MLLM as refiner")
            self.refiner = MistralRefinerwithMLLM(model_name=refine_pretrain_model.split('+')[0], max_new_tokens=128, **refine_pretrain_model_config).to(self.dtype)
            if refine_checkpoint_path is not None:
                self.refiner.load_checkpoint(refine_checkpoint_path)
        elif 'mistral' in refine_pretrain_model and refine_pretrain_model.split('+')[-1] == 'classname':
            print("Using Mistral with Classname as refiner")
            self.refiner = MistralRefinerwithClassname(model_name=refine_pretrain_model.split('+')[0], max_new_tokens=128, **refine_pretrain_model_config).to(self.dtype)
            if refine_checkpoint_path is not None:
                self.refiner.load_checkpoint(refine_checkpoint_path)
        elif 'llama' in refine_pretrain_model and refine_pretrain_model.split('+')[-1] == 'classname':
            print("Using Llama with Classname as refiner")
            self.refiner = MistralRefinerwithClassname(model_name=refine_pretrain_model.split('+')[0], max_new_tokens=128, **refine_pretrain_model_config).to(self.dtype)
            if refine_checkpoint_path is not None:
                self.refiner.load_checkpoint(refine_checkpoint_path)
        else:
            self.refiner = None

        if sentinel_pretrain_model is None:
            self.sentinel = None
        elif 'mistral' in sentinel_pretrain_model:
            print("Using Mistral as sentinel")
            self.sentinel = MistralCleaner(model_name=sentinel_pretrain_model, **sentinel_pretrain_model_config).to(self.dtype)
            if sentinel_checkpoint_path is not None:
                self.sentinel.load_checkpoint(sentinel_checkpoint_path)

    def to(self, *args, **kwargs):
        """
        Override to method to ensure the model and refiner are moved to the correct device.
        """
        super().to(*args, **kwargs)
        self.model.to(*args, **kwargs)
        if self.refiner is not None:
            self.refiner.to(*args, **kwargs)
        if self.sentinel is not None:
            self.sentinel.to(*args, **kwargs)
        return self

    def generate(self,
                 prompts: List[str],
                 **kwargs
                ):
        classnames = kwargs.get("classnames", None)
        use_cache = kwargs.get("use_cache", False)
        if use_cache:
            if self.sentinel is not None and self.refiner is not None:
                sentinel_kwargs = kwargs.get("sentinel_kwargs", {})
                cleaned_file = sentinel_kwargs.get("cleaned_file", None)
                cleaned_batch_size = sentinel_kwargs.get("cleaned_batch_size", 32)
                refiner_kwargs = kwargs.get("refiner_kwargs", {})
                refined_file = refiner_kwargs.get("refined_file", None)
                refine_batch_size = refiner_kwargs.get("refine_batch_size", 32)

                if cleaned_file is not None and refined_file is not None:
                    print(f"Loading cleaned prompts from {cleaned_file} and refined prompts from {refined_file}")
                    cleaned_df = pd.read_csv(cleaned_file, usecols=['input_text', 'cleaned_text'])
                    refined_df = pd.read_csv(refined_file, usecols=['input_text', 'refined_text'])
                    assert cleaned_df['cleaned_text'].equals(refined_df['input_text']), "Cleaned prompts and refined prompts do not match."
                    search_dict = dict(zip(cleaned_df['input_text'], refined_df['refined_text']))
                    for p in prompts:
                        if p not in search_dict:
                            raise ValueError(f"Prompt {p} not found in cleaned file {cleaned_file}.")
                    prompts = [search_dict[p] for p in prompts]
                elif cleaned_file is None and refined_file is None:
                    print("No cleaned or refined file provided, generating cleaned and refined prompts.")
                    prompts = self.sentinel.generate(prompts, **sentinel_kwargs)
                    prompts = self.refiner.generate(prompts, **refiner_kwargs)
                else:
                    raise ValueError("Both cleaned_file and refined_file must be provided or both must be None.")
            elif self.sentinel is not None:
                sentinel_kwargs = kwargs.get("sentinel_kwargs", {})
                cleaned_file = sentinel_kwargs.get("cleaned_file", None)
                cleaned_batch_size = sentinel_kwargs.get("cleaned_batch_size", 32)
                if cleaned_file is not None:
                    print(f"Loading cleaned prompts from {cleaned_file}")
                    cleaned_df = pd.read_csv(cleaned_file, usecols=['input_text', 'cleaned_text'])
                    for p in prompts:
                        if p not in cleaned_df['input_text'].values:
                            raise ValueError(f"Prompt {p} not found in cleaned file {cleaned_file}.")
                    search_dict = dict(zip(cleaned_df['input_text'], cleaned_df['cleaned_text']))
                    prompts = [search_dict[p] for p in prompts]
                else:
                    print("No cleaned file provided, generating cleaned prompts.")
                    prompts = self.sentinel.generate(prompts, **sentinel_kwargs)
            elif self.refiner is not None:
                refiner_kwargs = kwargs.get("refiner_kwargs", {})
                refined_file = refiner_kwargs.get("refined_file", None)
                refine_batch_size = refiner_kwargs.get("refine_batch_size", 32)
                if refined_file is not None:
                    print(f"Loading refined prompts from {refined_file}")
                    refined_df = pd.read_csv(refined_file, usecols=['input_text', 'refined_text'])
                    for p in prompts:
                        if p not in refined_df['input_text'].values:
                            raise ValueError(f"Prompt {p} not found in refined file {refined_file}.")
                    search_dict = dict(zip(refined_df['input_text'], refined_df['refined_text']))
                    prompts = [search_dict[p] for p in prompts]
                else:
                    print("No refined file provided, generating refined prompts.")
                    prompts = self.refiner.generate(prompts, **refiner_kwargs)
        else:
            if self.sentinel is not None:
                prompts = self.sentinel.generate(prompts, **kwargs.get("sentinel_kwargs", {}))
            if self.refiner is not None:
                if kwargs.get("save_refined", True):
                    refiner_kwargs = kwargs.get("refiner_kwargs", {})
                    refined_file = refiner_kwargs.get("refined_file", "refined_prompts.csv")
                    refined_prompts = self.refiner.generate(prompts, classnames=classnames, **kwargs.get("refiner_kwargs", {}))
                    df = pd.DataFrame({"input_text": prompts, "refined_text": refined_prompts})
                    if os.path.exists(refined_file):
                        df.to_csv(refined_file, mode='a', header=False, index=False)
                    else:
                        df.to_csv(refined_file, index=False)
                    prompts = refined_prompts
                else:
                    prompts = self.refiner.generate(prompts, classnames=classnames, **kwargs.get("refiner_kwargs", {}))

        if self.sd_pretrain_model == 'comat':
            images = self.model.predict(prompts, **kwargs)
        else:
            images = self.model(prompts, **kwargs)
        return images
    

        

        
                
                