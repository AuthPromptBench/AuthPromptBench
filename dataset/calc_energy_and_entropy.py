import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from typing import List, Union
from metrics.utils.load_benchmark import (
    DATA_FILE_PATH,
    PROJECT_ROOT,
    load_benchmark_data,
    load_refined_data
)

from lightning import seed_everything

from huggingface_hub import login

login(token="hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

def compute_batch_energy_entropy(texts, model, tokenizer, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    texts = [tokenizer.bos_token + text + tokenizer.eos_token for text in texts]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        
        attention_mask = inputs["attention_mask"][:, 1:]
        token_counts = attention_mask.sum(dim=1)


        selected_logits = logits.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        selected_logits = selected_logits * attention_mask
        energies = (-selected_logits.sum(dim=1) / token_counts).cpu().tolist()

        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-12)  # 防止log(0)
        entropies = (-(probs * log_probs).sum(dim=-1) * attention_mask).sum(dim=1)
        entropies = (entropies / token_counts).cpu().tolist()

    return energies, entropies

def process_texts(texts, model, tokenizer, device: str = "cuda" if torch.cuda.is_available() else "cpu", batch_size: int = 16):
    energies, entropies = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="calculating energy and entropy"):
        batch = texts[i:i+batch_size]
        batch_E, batch_H = compute_batch_energy_entropy(batch, model, tokenizer, device)
        for idx, (e, h) in enumerate(zip(batch_E, batch_H)):
            if not torch.isfinite(torch.tensor(e)) or torch.isnan(torch.tensor(h)):
                print(f"Anomalous value: Index={i+idx}, Energy={e}, Entropy={h}, Text={batch[idx]}, prompt={texts[i+idx]}")
        energies.extend(batch_E)
        entropies.extend(batch_H)

    print(f"Average energy: {sum(energies)/len(energies):.4f}, Average entropy: {sum(entropies)/len(entropies):.4f}")
    return energies, entropies

# 修改主函数支持自动识别文件类型
if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser(description="calculate energy and entropy for texts in a CSV or JSON file")

    parser.add_argument('--refiner_type', type=str, default=None, help='Type of refiner to use.')
    parser.add_argument('--model_name', type=str, required=True, help='Pretrained model name for energy and entropy calculation.')
    args = parser.parse_args()
    seed_everything(995)
    if args.refiner_type:
        data = load_refined_data(args.refiner_type)
        output_dir = f'data/benchmark/energy_and_entropy/{args.refiner_type}/{args.model_name}'
        print(f"Loaded {len(data)} refined prompts using refiner type: {args.refiner_type}")
        print(f"Output directory: {output_dir}")
    else:
        data = load_benchmark_data()
        output_dir = f'data/benchmark/energy_and_entropy/original/{args.model_name}'
        print(f"Loaded {len(data)} original prompts.")
        print(f"Output directory: {output_dir}")       

    prompts = [item['prompt'] for item in data]
    prompts = [p if p is not None else "" for p in prompts]

    if 'PAE' in args.refiner_type.upper():
        print("Detected PAE model, using special handling.")
        def convert_to_clear_prompt(prompt: str) -> str:
            """
            takes a prompt string that may contain segments in the format [text:start-end:weight] and converts it to a clear prompt by removing the metadata.
            """
            if isinstance(prompt, list):
                return [convert_to_clear_prompt(p) for p in prompt]

            s = prompt.replace("[ ", " [")
            out = ""
            i = 0
            L = len(s)
            while i < L:
                if s[i] != "[":
                    out += s[i]
                    i += 1
                    continue
                j = s.find("]", i + 1)
                if j == -1:
                    out += s[i + 1 :]
                    break
                inner = s[i + 1 : j]
                parts = inner.split(":")
                if len(parts) == 3:
                    out += parts[0]
                else:
                    out += inner
                i = j + 1
            if len(out) > 2000:
                print(f"warning: Processed prompt length is too long ({len(out)} characters), which may indicate an issue. Original prompt: {prompt}")
                return out[:2000]
            return out
        
        prompts = [convert_to_clear_prompt(p) for p in prompts]


    os.makedirs(output_dir, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    energies, entropies = process_texts(prompts, model, tokenizer, batch_size=16)

    # Save results
    import numpy as np
    np.savez_compressed(
        os.path.join(output_dir, 'energy_entropy.npz'),
        energies=np.array(energies),
        entropies=np.array(entropies)
    )
    print(f"Saved energy and entropy to {os.path.join(output_dir, 'energy_entropy.npz')}")
