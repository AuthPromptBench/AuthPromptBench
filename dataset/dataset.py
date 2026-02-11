import json
from PIL import Image
import os
# from petrel_client.client import Client
# from aoss_client.client import Client

import torch
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch import LightningDataModule
from torchvision import transforms
from PIL import Image
from datasets import load_dataset
import json
import random


def read_jsonl(save_path):
    ret_list = []
    with open(save_path, 'r') as f:
        for line in f:
            ret_list.append(json.loads(line))
    return ret_list

class Gan_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        training_prompts: str,
    ):
        if 'txt' in training_prompts:
            self.ann = list()
            with open(training_prompts, 'r') as f:
                for line in f:
                    self.ann.append(line.strip())
        elif 'jsonl' in training_prompts:
            self.ann = read_jsonl(training_prompts)
        elif 'json' in training_prompts:
            self.ann = json.load(open(training_prompts, 'r'))

        # self.client = Client('~/aoss.conf')

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        example = {}
        example['text'] = self.ann[index]['prompt']
        latent_path = self.ann[index]['file_path'] if not isinstance(self.ann[index]['file_path'], list) else random.choice(self.ann[index]['file_path'])
        latents = torch.load(latent_path)
        
        example['latents'] = latents
        
        # add another potential key in example
        eliminate_keys = ['prompt', 'file_path', 'image']
        for k in self.ann[index].keys():
            if k not in eliminate_keys:
                example[k] = self.ann[index][k]

        return example

class SimpleTextDataset(Dataset):
    def __init__(self, text):
        self.data = [text]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"text": self.data[idx]}
    

class GanDataModule(LightningDataModule):
    def __init__(
            self,
            train_batch_size: int,
            dataloader_num_workers: int,
            gan_loss: bool,
            training_prompts: str,
            resolution: int,
            center_crop: bool,
            image_folder: str = None,
            max_train_samples: int = None,
            validation_prompts_file: str = None,
            validation_prompts: str = "A man walking on street",
            
    ):
        super().__init__()
        self.save_hyperparameters()


    def setup(self, stage=None):
        if self.hparams.gan_loss:
            dataset = Gan_Dataset(self.hparams.training_prompts)
        elif self.hparams.training_prompts.endswith("txt"):
            dataset = load_dataset("text", data_files=dict(train=self.hparams.training_prompts))
        elif self.hparams.training_prompts.endswith("json"):
            dataset = load_dataset("json", data_files=dict(train=self.hparams.training_prompts))

        image_transforms = transforms.Compose(
            [
                transforms.Resize(self.hparams.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.hparams.resolution) if self.hparams.center_crop else transforms.RandomCrop(self.hparams.resolution),
                transforms.ToTensor()
            ]
        )

        def preprocess_train(instance):
            if 'file_name' in instance:
                filenames = instance.pop('file_name')
                images = [Image.open(os.path.join(self.hparams.image_folder, filename)).convert("RGB") for filename in filenames]
                images = [image_transforms(image) for image in images]
                instance['images'] = images
            return instance

        if self.hparams.gan_loss:
            self.train_dataset = dataset
        else:
            dataset["train"] = dataset["train"].shuffle()
            if self.hparams.max_train_samples is not None:
                dataset["train"] = dataset["train"].select(range(self.hparams.max_train_samples))
            # Set the training transforms
            self.train_dataset = dataset["train"].with_transform(preprocess_train)

        # If validation prompts are provided, create a validation dataset
        if self.hparams.validation_prompts_file:
            val_dataset = load_dataset("text", data_files=dict(validation=self.hparams.validation_prompts_file))
            self.val_dataset = val_dataset["validation"]
        elif self.hparams.validation_prompts:
            self.val_dataset = SimpleTextDataset(self.hparams.validation_prompts)


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.dataloader_num_workers,
        )

    def val_dataloader(self):
        if hasattr(self, 'val_dataset'):
            return DataLoader(
                self.val_dataset,
                batch_size=self.hparams.train_batch_size,
                num_workers=self.hparams.dataloader_num_workers,
            )
        else:
            return None
