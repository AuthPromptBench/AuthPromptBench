from typing import List, Dict, Any
import torch
from torch import nn

from .blip.caption_blip import Blip


class CaptionModelWrapper(torch.nn.Module):
    def __init__(
        self, 
        caption_model_names: List[str], 
        weights: List[float], 
        dtype: torch.dtype):
        super().__init__()

        self.dtype = dtype

        self.caption_model_dict = nn.ModuleDict()
        self.weights = {}
        for model, weight in zip(caption_model_names, weights):
            self.weights[model] = weight
            self.caption_model_dict[model] = self.load_model(model)

    def load_model(self, caption_model_name):
        if caption_model_name == 'Blip':
            return Blip('Salesforce/blip-image-captioning-large', self.dtype)
        raise NotImplementedError(f"Caption model {caption_model_name} is not implemented.")

    def forward(self, images, prompts, text_encoder=None, return_feature=False, step=-1, batch=None):
        caption_rewards = {}

        for model_name, model in self.caption_model_dict.items():
            if hasattr(model, 'score'):
                reward = model.score(images, prompts)
            else:
                raise NotImplementedError(f"Model {model_name} does not have a score method.")
            caption_rewards[model_name] = reward

        # Combine rewards from all models
        combined_reward = sum(self.weights[model_name] * caption_rewards[model_name] for model_name in caption_rewards.keys())
        caption_rewards['combined'] = combined_reward
        return caption_rewards
