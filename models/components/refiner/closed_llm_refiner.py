import requests
from typing import Callable, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock



class OpenRouterIntentPreservingRefiner:
    SYSTEM_PROMPT = (
        "You are an expert text-to-image prompt editor. "
        "Intent preservation is more important than brevity. "
        "Preserve all explicitly stated entities, quantities, actions, "
        "attributes, materials, relations, scenes, and styles. "
        "Do not add unsupported details or omit meaningful content. "
        "Return only the refined prompt."
    )

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        max_tokens: int = 256,
        temperature: float = 0.2,
        timeout: int = 120,
        retries: int = 3,
        retry_sleep: float = 2.0,
        concurrency: int = 50,
        request_interval: float = 0.0,
        incremental_save_every: int = 0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep = retry_sleep
        self.concurrency = concurrency
        self.request_interval = request_interval
        self.incremental_save_every = incremental_save_every
        self._request_lock = Lock()
        self._last_request_started = 0.0

    def _throttle_request_start(self) -> None:
        if self.request_interval <= 0:
            return
        import time

        with self._request_lock:
            now = time.monotonic()
            wait_seconds = self.request_interval - (now - self._last_request_started)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_request_started = time.monotonic()

    def refine_prompt(self, prompt: str) -> str:
        import time

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        last_error = None
        for attempt in range(self.retries + 1):
            try:
                self._throttle_request_start()
                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    return self._clean_response(data["choices"][0]["message"]["content"] or "")
                last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(self.retry_sleep)
        raise RuntimeError(f"OpenRouter refinement failed: {last_error}")

    def _clean_response(self, text: str) -> str:
        refined_prompt = text.strip().strip('"')
        if refined_prompt.lower().startswith("assistant"):
            refined_prompt = refined_prompt[len("assistant"):].strip()
        for prefix in ("Optimized prompt:", "Refined prompt:", "Prompt:"):
            if refined_prompt.lower().startswith(prefix.lower()):
                refined_prompt = refined_prompt[len(prefix):].strip()
        return "\n".join(line.strip() for line in refined_prompt.splitlines() if line.strip())

    def generate_and_save(self, plain_texts: List[str], save_function: Callable):
        refined_texts = [None] * len(plain_texts)
        workers = max(1, self.concurrency)
        completed = 0
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.refine_prompt, prompt): index
                for index, prompt in enumerate(plain_texts)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Refining prompts", ncols=80):
                index = futures[future]
                try:
                    refined_texts[index] = future.result()
                except Exception as exc:
                    failures += 1
                    print(f"OpenRouter refinement failed for index {index}: {exc}", flush=True)
                    refined_texts[index] = ""
                completed += 1
                if self.incremental_save_every and completed % self.incremental_save_every == 0:
                    save_function(refined_texts)
                    print(
                        f"Incremental save after {completed}/{len(plain_texts)} completions "
                        f"(failures={failures}).",
                        flush=True,
                    )
        save_function(refined_texts)

        
