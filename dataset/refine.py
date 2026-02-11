import os
import json
from typing import List
from models.components.refiner.refiner import RefinerModel
from models.components.refiner.pae_refiner import PAE_refiner
# from models.components.refiner.llmrefiner import MistralRefiner, MistralRefinerwithLM, MistralRefinerwithNLP, MistralRefinerwithMLLM

from metrics.utils.load_benchmark import (
    DATA_FILE_PATH,
    PROJECT_ROOT,
    load_benchmark_data
)

from lightning import seed_everything

REFINED_DATA_DIR = os.path.join(PROJECT_ROOT, "data/benchmark/refined_prompts")
os.makedirs(REFINED_DATA_DIR, exist_ok=True)

def refine_prompts_and_save(
        refiner_type: str,
        **kwargs
        ):    
    datas = load_benchmark_data(DATA_FILE_PATH)
    prompts = [item['prompt'] for item in datas]
    print(f"Loaded {len(prompts)} prompts for refinement.")

    match refiner_type:
        case "promptist":
            refiner = RefinerModel(pretrained_model_name="microsoft/Promptist")
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, "promptist.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = refiner_type
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)
        case "pag":
            refiner = RefinerModel(pretrained_model_name="pag")
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, "pag.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = refiner_type
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)
        case "PAE":
            model_name = kwargs.get("model_name")
            print(f"Using PAE model: {model_name}")
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/PAE/{model_name}/actor_step3000.pt")
            refiner = PAE_refiner(ckpt_path=ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"pae_{model_name}.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"PAE_{model_name}"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'mistral3_grpo':
            from models.components.refiner.llmrefiner import Mistral3Refiner
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/mistral3_grpo/version_0/refiner-epoch=00-train_loss=-0.03-v2.ckpt")
            refiner = Mistral3Refiner(model_name="mistralai/Ministral-3-8B-Instruct-2512", use_fp8=False)
            refiner.load_checkpoint(ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"mistral3_grpo.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"mistral3_grpo"
                    new_data['refiner_checkpoint'] = ckpt_path
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)
        
        case 'mistral3_sft':
            from models.components.refiner.llmrefiner import Mistral3Refiner
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/mistral3_sft/version_0/refiner-epoch=02-train_loss=1.44.ckpt")
            refiner = Mistral3Refiner(model_name="mistralai/Ministral-3-8B-Instruct-2512", use_fp8=False)
            refiner.load_checkpoint(ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"mistral3_sft.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"mistral3_sft"
                    new_data['refiner_checkpoint'] = ckpt_path
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'mistral3':
            from models.components.refiner.llmrefiner import Mistral3Refiner
            refiner = Mistral3Refiner(model_name="mistralai/Ministral-3-8B-Instruct-2512", use_fp8=False, use_lora=False)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"mistral3.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"mistral3"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'llama3_grpo':
            from models.components.refiner.llmrefiner import AutoModelLLMRefiner
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/llama3_grpo/version_0/refiner-epoch=00-train_loss=-0.00.ckpt")
            refiner = AutoModelLLMRefiner(model_name="meta-llama/Llama-3.1-8B-Instruct")
            refiner.load_checkpoint(ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"llama3_grpo.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"llama3_grpo"
                    new_data['refiner_checkpoint'] = ckpt_path
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'llama3_sft':
            from models.components.refiner.llmrefiner import AutoModelLLMRefiner
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/llama3_sft/version_0/refiner-epoch=02-train_loss=0.93.ckpt")
            refiner = AutoModelLLMRefiner(model_name="meta-llama/Llama-3.1-8B-Instruct")
            refiner.load_checkpoint(ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"llama3_sft.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"llama3_sft"
                    new_data['refiner_checkpoint'] = ckpt_path
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)
        
        case 'llama3':
            from models.components.refiner.llmrefiner import AutoModelLLMRefiner
            refiner = AutoModelLLMRefiner(model_name="meta-llama/Llama-3.1-8B-Instruct", use_lora=False)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"llama3.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"llama3"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'qwen3_grpo':
            from models.components.refiner.llmrefiner import Qwen3LLMRefiner
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/qwen3_grpo/version_0/refiner-epoch=00-train_loss=-0.01-v1.ckpt")
            refiner = Qwen3LLMRefiner(model_name="Qwen/Qwen3-8B", use_lora=True)
            refiner.load_checkpoint(ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"qwen3_grpo.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"qwen3_grpo"
                    new_data['refiner_checkpoint'] = ckpt_path
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'qwen3_sft':
            from models.components.refiner.llmrefiner import Qwen3LLMRefiner
            ckpt_path = os.path.join(PROJECT_ROOT, f"checkpoints/qwen3_sft/version_0/refiner-epoch=02-train_loss=1.33.ckpt")
            refiner = Qwen3LLMRefiner(model_name="Qwen/Qwen3-8B", use_lora=True)
            refiner.load_checkpoint(ckpt_path)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"qwen3_sft.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"qwen3_sft"
                    new_data['refiner_checkpoint'] = ckpt_path
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'qwen3':
            from models.components.refiner.llmrefiner import Qwen3LLMRefiner
            refiner = Qwen3LLMRefiner(model_name="Qwen/Qwen3-8B", use_lora=False)
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"qwen3.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"qwen3"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'gpt-4o-mini':
            from models.components.refiner.closed_llm_refiner import GPTRefiner
            api_key = ''
            refiner = GPTRefiner(api_key=api_key, model="openai/gpt-4o-mini")
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"gpt-4o-mini.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"gpt-4o-mini"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'gemini-2.0':
            from models.components.refiner.closed_llm_refiner import GPTRefiner
            api_key = ''
            refiner = GPTRefiner(api_key=api_key, model="google/gemini-2.0-flash-lite-001")
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"gemini-2.0.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"gemini-2.0"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'grok-3':
            from models.components.refiner.closed_llm_refiner import GPTRefiner
            api_key = ''
            refiner = GPTRefiner(api_key=api_key, model="x-ai/grok-3-mini")
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"grok-3.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"grok-3"
                    refined_datas.append(new_data)

                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case 'deepseek-3.1':
            from models.components.refiner.closed_llm_refiner import GPTRefiner
            api_key = ''
            refiner = GPTRefiner(api_key=api_key, model="deepseek/deepseek-chat-v3.1")
            def save_function(refined_texts: List[str]):
                assert len(prompts) == len(refined_texts)
                refined_data_path = os.path.join(REFINED_DATA_DIR, f"deepseek-3.1.json")
                refined_datas = []
                for original_data, refined_text in zip(datas, refined_texts):
                    new_data = original_data.copy()
                    new_data['refined_prompt'] = refined_text
                    new_data['refiner_type'] = f"deepseek-3.1"
                    refined_datas.append(new_data)
                with open(refined_data_path, 'w', encoding='utf-8') as f:
                    json.dump(refined_datas, f, ensure_ascii=False, indent=4)
            refiner.generate_and_save(prompts, save_function)

        case _:
            raise ValueError(f"Unknown refiner type: {refiner_type}")
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--refiner_type', type=str, required=True, help='Type of refiner to use.')
    parser.add_argument('--model_name', type=str, required=False, help='Model name for PAE refiner.')
    args = parser.parse_args()
    seed_everything(995)
    print(f"Using refiner type: {args.refiner_type}")
    refine_prompts_and_save(refiner_type=args.refiner_type, model_name=args.model_name)
            
