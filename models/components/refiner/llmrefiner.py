"""Local prompt editors used by the public refinement entry point."""

from __future__ import annotations

from collections.abc import Callable, Sequence


SYSTEM_PROMPT = (
    "You are an expert text-to-image prompt editor. Preserve every explicitly stated "
    "entity, quantity, action, attribute, material, relation, scene, and style. "
    "Do not invent unsupported details or remove meaningful content. Return only the edited prompt."
)
INSTRUCTION = "Rewrite the input as a clear, executable text-to-image prompt without changing its intent."


class IntentPreservingAutoModelLLMRefiner:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        use_lora: bool = False,
        use_fp8: bool = False,
        **_: object,
    ) -> None:
        del use_fp8
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "v_proj"],
                ),
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)
        if not state:
            raise ValueError(f"Checkpoint has no weights: {checkpoint_path}")
        self.model.load_state_dict(state, strict=False)

    def _render(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUCTION}\n\nInput:\n{prompt}"},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts: Sequence[str], batch_size: int = 4, **_: object) -> list[str]:
        import torch

        outputs: list[str] = []
        self.model.eval()
        for start in range(0, len(prompts), batch_size):
            batch = list(prompts[start : start + batch_size])
            rendered = [self._render(prompt) for prompt in batch]
            tokens = self.tokenizer(rendered, return_tensors="pt", padding=True).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    **tokens,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = generated[:, tokens["input_ids"].shape[1] :]
            outputs.extend(text.strip() for text in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    def generate_and_save(self, prompts: Sequence[str], save_function: Callable[[list[str]], None]) -> None:
        save_function(self.generate(prompts))


class IntentPreservingQwen3LLMRefiner(IntentPreservingAutoModelLLMRefiner):
    pass


class IntentPreservingMistral3Refiner(IntentPreservingAutoModelLLMRefiner):
    pass
