import json
import torch
from torch.utils.data import Dataset
import difflib

def format_prompt(sample):
    return f"### Instruction:\n{sample['instruction']}\n\n### Input:\n{sample['input']}\n\n### Response:\n{sample['output']}"
class PromptDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prompt = format_prompt(sample)
        tokenized = self.tokenizer(prompt, truncation=True, max_length=self.max_length, padding="max_length", return_tensors="pt")
        tokenized["labels"] = tokenized["input_ids"].clone()
        return {k: v.squeeze(0) for k, v in tokenized.items()}
    
class PromptAndImageDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            self.samples = json.load(f)
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
    
class PromptDatasetForGRPO(Dataset):
    def __init__(self, path):
        self.data = [json.loads(line) for line in open(path)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        return {
            "input": example["input"],
            "instruction": example["instruction"],
            "chosen": example["refined_prompt_best"],
            "rejected": example["refined_prompt_worst"]
        }
    
    
            
        
    

