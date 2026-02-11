from typing import Union, List, Callable
import os
from tqdm import tqdm
from torch import nn
from transformers import (
    GPT2LMHeadModel, GPT2TokenizerFast,
    AutoModelForCausalLM, AutoConfig
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))


CACHE_FILE = os.path.join(os.path.dirname(__file__), "../../../data/refiner_cache/cache.tsv")

def load_prompt_refiner(pretrained_model_name: str = "microsoft/Promptist"):
    if pretrained_model_name == "microsoft/Promptist":
        prompter_model = GPT2LMHeadModel.from_pretrained(pretrained_model_name)
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return prompter_model, tokenizer
    elif pretrained_model_name == "pag":
        config = AutoConfig.from_pretrained('gpt2')
        prompter_model = AutoModelForCausalLM.from_pretrained('gpt2', config=config)
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints/pag")
        print(f"Loading PAG checkpoint from {ckpt_dir}")
        _prompter_model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
        msg = prompter_model.load_state_dict(_prompter_model.state_dict(), strict=False)
        print(f"Loaded PAG checkpoint with message: {msg}")
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return prompter_model, tokenizer
    else:
        raise ValueError(f"Unknown pretrained model name: {pretrained_model_name}")

def load_cache(cache_file: str = CACHE_FILE):
    if not os.path.exists(cache_file):
        return {}
    
    cache = {}
    with open(cache_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('\t')
            cache[key] = value
    return cache

class RefinerModel(nn.Module):
    def __init__(self, pretrained_model_name: str):
        super(RefinerModel, self).__init__()
        self.model, self.tokenizer = load_prompt_refiner(pretrained_model_name)

    def generate(self, plain_text: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        device = self.model.device
        # 处理单个文本或文本列表
        if isinstance(plain_text, str):
            plain_texts = [plain_text]
            single_input = True
        else:
            plain_texts = plain_text
            single_input = False
        
        # 为所有文本添加"Rephrase:"后缀
        input_texts = []
        for text in plain_texts:
            if len(text.strip()) >= 2000:
                if '. ' in text:
                    text = text.split('. ')[0]
                    text = text + '.'
                elif '! ' in text:
                    text = text.split('! ')[0]
                    text = text + '!'
                elif ', ' in text:
                    text = text.split(', ')[0]
                    text = text + '.'
                else:
                    text = text[:500]
            if len(text.strip()) >= 2000:
                text = text[:500]
            input_texts.append(text.strip() + " Rephrase:")


        results = []
        for text in tqdm(input_texts, desc="Generating texts", ncols=80):
            # Tokenize输入文本
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(device)
            eos_id = self.tokenizer.eos_token_id

            # 生成输出
            outputs = self.model.generate(
                input_ids, 
                do_sample=False,
                num_beams=8, 
                max_new_tokens=128,
                num_return_sequences=8, 
                eos_token_id=eos_id, 
                pad_token_id=eos_id, 
                length_penalty=-1.0
            )
            
            # 解码输出
            output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # 处理生成结果
            res = output_texts[0].replace(text, "").strip()
            results.append(res)

        # # 批量tokenize
        # input_ids = self.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).input_ids
        # eos_id = self.tokenizer.eos_token_id

        # # 将输入数据移动到GPU
        # input_ids = input_ids.to(device)

        # # 批量生成
        # outputs = self.model.generate(
        #     input_ids, 
        #     do_sample=False, 
        #     max_new_tokens=128, 
        #     num_beams=8, 
        #     num_return_sequences=8, 
        #     eos_token_id=eos_id, 
        #     pad_token_id=eos_id, 
        #     length_penalty=-1.0
        # )
        
        # # 解码输出
        # output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # # 处理生成结果
        # results = []
        # for i, original_text in enumerate(plain_texts):
        #     # 每个输入对应8个输出（num_return_sequences=8）
        #     batch_outputs = output_texts[i*8:(i+1)*8]
        #     # 取第一个结果并清理
        #     res = ''
        #     j = 0
        #     while res == '':
        #         res = batch_outputs[j].replace(original_text + " Rephrase:", "").strip()
        #         j += 1
        #         if j >= len(batch_outputs):
        #             raise ValueError(f"No valid output found for input: {original_text}")
            
        #     results.append(res)
        
        # 如果输入是单个字符串，返回单个结果
        return results[0] if single_input else results
    
    def generate_and_save(self, plain_texts: List[str], save_function: Callable):
        refined_texts = self.generate(plain_texts)
        save_function(refined_texts)
        
        

    
