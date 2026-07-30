from __future__ import annotations

import math
import time
from threading import Lock

from epistemic_uq.backends.base import ModelBackend, build_generation
from epistemic_uq.errors import BackendError
from epistemic_uq.schemas import BackendCapabilities, ModelRequest, TokenProbability, Usage
from epistemic_uq.utils import elapsed_ms


class HuggingFaceBackend(ModelBackend):
    _cache: dict[str, tuple[object, object]] = {}
    _lock = Lock()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            logprobs=True,
            seed=True,
            multi_sample=True,
            forced_confidence=True,
            streaming=False,
            local=True,
        )

    def _load(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError("Install epistemic-uq[local] to use HuggingFaceBackend") from exc
        key = f"{self.config.model}:{self.config.options.get('revision', 'main')}"
        with self._lock:
            if key not in self._cache:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.model,
                    revision=self.config.options.get("revision"),
                    trust_remote_code=bool(self.config.options.get("trust_remote_code", False)),
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.config.model,
                    revision=self.config.options.get("revision"),
                    trust_remote_code=bool(self.config.options.get("trust_remote_code", False)),
                    device_map=self.config.options.get("device_map", "auto"),
                    torch_dtype=self.config.options.get("torch_dtype", "auto"),
                )
                model.eval()
                self._cache[key] = (tokenizer, model)
        return self._cache[key]

    def _format_prompt(self, tokenizer, request: ModelRequest) -> str:
        messages = []
        if request.prompt.system:
            messages.append({"role": "system", "content": request.prompt.system})
        messages.append({"role": "user", "content": request.prompt.user})
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n\n".join(item["content"] for item in messages)

    def _generate(self, request: ModelRequest):
        try:
            import torch
        except ImportError as exc:
            raise BackendError("Install epistemic-uq[local] to use HuggingFaceBackend") from exc
        tokenizer, model = self._load()
        prompt_text = self._format_prompt(tokenizer, request)
        inputs = tokenizer(prompt_text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(request.seed)
        do_sample = request.temperature > 0.0
        generation_kwargs = {
            "max_new_tokens": request.max_tokens,
            "do_sample": do_sample,
            "temperature": max(request.temperature, 1e-6),
            "top_p": request.top_p,
            "return_dict_in_generate": True,
            "output_scores": request.logprobs,
            "pad_token_id": tokenizer.eos_token_id,
        }
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)
        prompt_length = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0, prompt_length:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        token_probabilities: list[TokenProbability] = []
        cursor = 0
        if request.logprobs and output.scores:
            for position, (token_id, logits) in enumerate(zip(generated_ids.tolist(), output.scores, strict=False)):
                log_probs = torch.log_softmax(logits[0], dim=-1)
                logprob = float(log_probs[token_id].item())
                token = tokenizer.decode([token_id], skip_special_tokens=False)
                start_char = cursor
                cursor += len(token)
                token_probabilities.append(
                    TokenProbability(
                        token=token,
                        logprob=logprob,
                        probability=math.exp(logprob),
                        position=position,
                        start_char=start_char,
                        end_char=cursor,
                    )
                )
        prompt_tokens = int(prompt_length)
        completion_tokens = int(generated_ids.shape[0])
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            estimated_cost=0.0,
        )
        return build_generation(
            request=request,
            backend_id=self.config.backend_id,
            text=text,
            latency_ms=elapsed_ms(started),
            finish_reason="stop",
            token_probabilities=token_probabilities,
            usage=usage,
            raw_response={"generated_token_ids": generated_ids.tolist()},
            reproducibility={
                "backend_type": self.config.backend_type.value,
                "model": self.config.model,
                "revision": self.config.options.get("revision"),
                "seed": request.seed,
                "generation_parameters": generation_kwargs,
            },
        )
