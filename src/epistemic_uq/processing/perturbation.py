from __future__ import annotations

import re
from collections.abc import Callable

from epistemic_uq.schemas import CanonicalEvaluationUnit, Perturbation, PerturbationFamily, Prompt
from epistemic_uq.utils import deterministic_random, stable_hash


class PerturbationEngine:
    def __init__(self, seed: int = 1729) -> None:
        self.seed = seed
        self.transforms: dict[str, Callable[[str, CanonicalEvaluationUnit], str]] = {
            "instruction_order": self._instruction_order,
            "formatting": self._formatting,
            "task_framing": self._task_framing,
            "lexical": self._lexical,
        }

    def build_family(
        self,
        example: CanonicalEvaluationUnit,
        baseline: Prompt,
        enabled: list[str] | tuple[str, ...],
        max_variants: int,
    ) -> PerturbationFamily:
        variants: list[Perturbation] = []
        for name in enabled:
            transform = self.transforms.get(name)
            if transform is None:
                raise ValueError(f"Unknown perturbation transform {name}")
            text = transform(baseline.user, example)
            if text == baseline.user:
                continue
            prompt = baseline.model_copy(
                update={
                    "user": text,
                    "template_id": f"{baseline.template_id}:{name}",
                    "variables": {**baseline.variables, "perturbation": name},
                }
            )
            variants.append(
                Perturbation(
                    perturbation_id=stable_hash({"example": example.example_id, "transform": name, "text": text})[:20],
                    transform=name,
                    prompt=prompt,
                    parameters={"seed": self.seed},
                )
            )
            if len(variants) >= max_variants:
                break
        return PerturbationFamily(example_id=example.example_id, baseline=baseline, variants=tuple(variants))

    def _instruction_order(self, text: str, example: CanonicalEvaluationUnit) -> str:
        segments = [segment.strip() for segment in re.split(r"\n{2,}", text) if segment.strip()]
        if len(segments) < 2:
            format_hint = example.expected_format or "a concise answer"
            return f"Provide {format_hint}.\n\n{text}"
        return "\n\n".join(segments[1:] + segments[:1])

    def _formatting(self, text: str, example: CanonicalEvaluationUnit) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        format_hint = example.expected_format or "plain text"
        return f"Task input:\n{compact}\n\nRequired response format:\n{format_hint}"

    def _task_framing(self, text: str, example: CanonicalEvaluationUnit) -> str:
        frames = {
            "question_answering": "Answer the following question using only the information and reasoning needed for the answer.",
            "classification": "Assign the single best valid label to the following input.",
            "extraction": "Extract the requested information from the following input without adding unsupported content.",
            "structured": "Produce a valid structured response matching the required format for the following input.",
        }
        return f"{frames[example.task_type.value]}\n\n{text}"

    def _lexical(self, text: str, example: CanonicalEvaluationUnit) -> str:
        rng = deterministic_random(self.seed, example.example_id)
        replacements = [
            (r"\bprovide\b", ("return", "give")),
            (r"\bdetermine\b", ("identify", "find")),
            (r"\bchoose\b", ("select", "pick")),
            (r"\banswer\b", ("response", "result")),
            (r"\bfollowing\b", ("given", "next")),
        ]
        result = text
        for pattern, candidates in replacements:
            replacement = candidates[rng.randrange(len(candidates))]
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if result == text:
            result = f"Work on this task carefully and return the requested result.\n\n{text}"
        return result
