import torch
from typing import List
from tqdm import tqdm
import os
from itertools import batched
from lightning.pytorch import LightningModule
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

class LLMCleaner(object):
    def generate_and_save_to_csv(self, plain_texts: List[str], csv_file: str, batch_size: int = 32, **kwargs):
        import csv
        results = []
        for batch in tqdm(batched(plain_texts, batch_size), desc="Generating texts", ncols=80, total=(len(plain_texts) + batch_size - 1) // batch_size):
            batch_results = self.generate(batch, **kwargs)
            results.extend(batch_results)
            if os.path.exists(csv_file):
                mode = 'a'
            else:
                mode = 'w'
            if not os.path.exists(os.path.dirname(csv_file)):
                os.makedirs(os.path.dirname(csv_file))
            with open(csv_file, mode, newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                if mode == 'w':
                    csv_writer.writerow(['input_text', 'cleaned_text'])
                for input_text, cleaned_text in zip(batch, batch_results):
                    csv_writer.writerow([input_text, cleaned_text])
        return results

class MistralCleaner(LightningModule, LLMCleaner):
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", lr=2e-4, use_lora=True):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto"
        )

        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
        

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        self.log("train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
        self.log("val_loss", outputs.loss)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
    
    def state_dict(self):
        # Only keep trainable parameters in the state dict
        state = super().state_dict()
        return {k: v for k, v in state.items() if v.requires_grad}
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint into the model"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'], strict=False)
        print(f"Checkpoint loaded from {checkpoint_path}")
    

    @torch.no_grad()
    def generate(self, inputs, instruction="Fix adversarial tokens and truncate toxic endings. Output only the cleaned text without explanation:", **kwargs):
        """
        参数:
            inputs: str 或 List[str]
            **kwargs: 传递给生成的额外参数
        返回:
            List[str] 或 str（与输入类型相同）
        """
        single_input = False
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True

        # 改进提示词格式，要求直接输出清理后的文本
        prompts = [
            f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
            for inp in inputs
        ]

        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        for k, v in tokenized.items():
            if isinstance(v, torch.Tensor):
                tokenized[k] = v.to(self.device)

        # 使用 kwargs 中的参数
        generation_kwargs = {
            'input_ids': tokenized["input_ids"],
            'attention_mask': tokenized["attention_mask"],
            'max_new_tokens': kwargs.get('max_new_tokens', 80),
            'do_sample': kwargs.get('do_sample', True),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'pad_token_id': self.tokenizer.eos_token_id
        }
        
        # 只保留非None的参数
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        output_ids = self.model.generate(**generation_kwargs)

        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # 后处理：提取并清理响应
        responses = []
        for prompt, text in zip(prompts, decoded):
            # 移除原始提示词
            gen = text.replace(prompt, "").strip()
            
            # 如果生成的文本包含 "### Input:" 说明模型重复了模板
            if "### Input:" in gen:
                # 只取第一个 "### Input:" 之前的内容
                gen = gen.split("### Input:")[0].strip()
            
            # 移除常见的解释性前缀
            cleanup_patterns = [
                "This description contains potential adversarial tokens and a toxic ending. Here's the cleaned-up version:",
                "Here's the cleaned version:",
                "Cleaned version:",
                "Here's the fixed version:",
                "Fixed version:",
                "The cleaned text is:",
                "Output:",
                "Response:",
            ]
            
            for pattern in cleanup_patterns:
                if gen.startswith(pattern):
                    gen = gen[len(pattern):].strip()
                    break
            
            # 移除括号中的解释性内容
            import re

            # 通用方法：移除所有包含解释性关键词的括号内容
            def clean_explanatory_content(text):
                # 定义解释性关键词
                explanatory_keywords = [
                    'removed', 'fixed', 'cleaned', 'eliminated', 'truncated', 'modified',
                    'toxic', 'ending', 'adversarial', 'tokens', 'content', 'description',
                    'started', 'excluded', 'as it is', 'could be considered', 'not related',
                ]
                
                # 移除包含任何关键词的圆括号内容
                for keyword in explanatory_keywords:
                    pattern = rf'\([^)]*{re.escape(keyword)}[^)]*\)'
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                
                # 移除包含任何关键词的方括号内容
                for keyword in explanatory_keywords:
                    pattern = rf'\[[^\]]*{re.escape(keyword)}[^\]]*\]'
                    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                
                # 移除包含引号的括号内容
                text = re.sub(r'\([^)]*"[^"]*"[^)]*\)', '', text)
                text = re.sub(r'\[[^\]]*"[^"]*"[^\]]*\]', '', text)
                
                # 移除包含特殊符号的括号内容
                special_symbols = ['➡️', '→', '⬇️', '⬆️', '⭐', '🔥']
                for symbol in special_symbols:
                    text = re.sub(rf'\([^)]*{re.escape(symbol)}[^)]*\)', '', text)
                    text = re.sub(rf'\[[^\]]*{re.escape(symbol)}[^\]]*\]', '', text)
                
                # 处理句尾只有左括号的情况：移除最后一个左括号及其后的所有内容
                if '(' in text and text.count('(') > text.count(')'):
                    # 找到最后一个未配对的左括号位置
                    open_count = 0
                    last_unmatched_open = -1
                    for i, char in enumerate(text):
                        if char == '(':
                            open_count += 1
                            if open_count > text[:i+1].count(')'):
                                last_unmatched_open = i
                        elif char == ')':
                            open_count -= 1
                    
                    if last_unmatched_open != -1:
                        text = text[:last_unmatched_open].strip()
                
                # 处理句尾只有右括号的情况：移除第一个未配对的右括号及其后的所有内容
                if ')' in text and text.count(')') > text.count('('):
                    # 找到第一个未配对的右括号位置
                    open_count = 0
                    first_unmatched_close = -1
                    for i, char in enumerate(text):
                        if char == '(':
                            open_count += 1
                        elif char == ')':
                            open_count -= 1
                            if open_count < 0 and first_unmatched_close == -1:
                                first_unmatched_close = i
                                break
                    
                    if first_unmatched_close != -1:
                        text = text[:first_unmatched_close].strip()
                
                # 处理方括号的情况
                # 移除句尾只有左方括号之后的所有内容
                if '[' in text and text.count('[') > text.count(']'):
                    # 找到最后一个未配对的左方括号位置
                    open_count = 0
                    last_unmatched_open = -1
                    for i, char in enumerate(text):
                        if char == '[':
                            open_count += 1
                            if open_count > text[:i+1].count(']'):
                                last_unmatched_open = i
                        elif char == ']':
                            open_count -= 1
                    
                    if last_unmatched_open != -1:
                        text = text[:last_unmatched_open].strip()
                
                # 移除句尾只有右方括号之后的所有内容
                if ']' in text and text.count(']') > text.count('['):
                    # 找到第一个未配对的右方括号位置
                    open_count = 0
                    first_unmatched_close = -1
                    for i, char in enumerate(text):
                        if char == '[':
                            open_count += 1
                        elif char == ']':
                            open_count -= 1
                            if open_count < 0 and first_unmatched_close == -1:
                                first_unmatched_close = i
                                break
                    
                    if first_unmatched_close != -1:
                        text = text[:first_unmatched_close].strip()
                
                return text

            # 应用清理函数
            gen = clean_explanatory_content(gen)

            # 移除任何剩余的空括号
            gen = re.sub(r'\(\s*\)', '', gen)
            gen = re.sub(r'\[\s*\]', '', gen)

            # 清理连续的空格和标点
            gen = re.sub(r'\s+', ' ', gen).strip()
            gen = re.sub(r'^[,.\s]+|[,.\s]+$', '', gen).strip()

            # 如果包含换行符，只取第一行（除非是明显的多行描述）
            lines = gen.split('\n')
            if len(lines) > 1:
                # 如果第一行看起来是完整的，只取第一行
                first_line = lines[0].strip()
                if len(first_line) > 10 and (first_line.endswith('.') or first_line.endswith(',') or len(first_line) > 50):
                    gen = first_line
                else:
                    # 否则合并所有非空行
                    gen = ' '.join([line.strip() for line in lines if line.strip()])
            
            # 移除多余的空格
            gen = ' '.join(gen.split())
            
            responses.append(gen)

        return responses[0] if single_input else responses
