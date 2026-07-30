from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

from epistemic_uq.processing.normalization import normalize_text
from epistemic_uq.schemas import ExtractedAnswer, SemanticCluster
from epistemic_uq.utils import stable_hash


NEGATION_PAIRS = {
    "yes": "no",
    "no": "yes",
    "true": "false",
    "false": "true",
    "correct": "incorrect",
    "incorrect": "correct",
    "supported": "unsupported",
    "unsupported": "supported",
    "positive": "negative",
    "negative": "positive",
}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root
        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root
        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


@dataclass(frozen=True)
class SemanticConfig:
    embedding_model: str | None = None
    cosine_threshold: float = 0.82
    numeric_absolute_tolerance: float = 1e-6
    numeric_relative_tolerance: float = 1e-6


class SemanticAdjudicator:
    def __init__(self, config: SemanticConfig | None = None) -> None:
        self.config = config or SemanticConfig()
        self._encoder = None

    def _numeric(self, value: str) -> Decimal | None:
        cleaned = value.strip().replace(",", "").removesuffix("%").strip()
        try:
            number = Decimal(cleaned)
        except InvalidOperation:
            return None
        if value.strip().endswith("%"):
            number /= Decimal(100)
        return number

    def _embedding_encoder(self):
        if self.config.embedding_model is None:
            return None
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.config.embedding_model)
        return self._encoder

    def equivalent(self, left: ExtractedAnswer, right: ExtractedAnswer) -> bool:
        if left.canonical == right.canonical:
            return True
        normalized_left = normalize_text(left.raw, remove_articles=True)
        normalized_right = normalize_text(right.raw, remove_articles=True)
        if normalized_left == normalized_right:
            return True
        left_number = self._numeric(left.raw)
        right_number = self._numeric(right.raw)
        if left_number is not None and right_number is not None:
            return math.isclose(
                float(left_number),
                float(right_number),
                abs_tol=self.config.numeric_absolute_tolerance,
                rel_tol=self.config.numeric_relative_tolerance,
            )
        encoder = self._embedding_encoder()
        if encoder is None:
            return False
        vectors = encoder.encode([left.raw, right.raw], normalize_embeddings=True)
        similarity = float(np.dot(vectors[0], vectors[1]))
        return similarity >= self.config.cosine_threshold

    def contradictory(self, left: ExtractedAnswer, right: ExtractedAnswer) -> bool:
        left_value = normalize_text(left.raw)
        right_value = normalize_text(right.raw)
        if NEGATION_PAIRS.get(left_value) == right_value:
            return True
        left_number = self._numeric(left.raw)
        right_number = self._numeric(right.raw)
        if left_number is not None and right_number is not None:
            magnitude = max(abs(float(left_number)), abs(float(right_number)), 1.0)
            return abs(float(left_number - right_number)) / magnitude > 0.5
        left_negated = bool(re.search(r"\b(?:not|never|no|false|incorrect)\b", left_value))
        right_negated = bool(re.search(r"\b(?:not|never|no|false|incorrect)\b", right_value))
        shared = set(left_value.split()) & set(right_value.split())
        return left_negated != right_negated and len(shared) >= 2

    def cluster(
        self,
        answers: tuple[ExtractedAnswer, ...],
        generation_ids: tuple[str, ...],
    ) -> tuple[SemanticCluster, ...]:
        if len(answers) != len(generation_ids):
            raise ValueError("Answers and generation identifiers must align")
        if not answers:
            return ()
        union_find = UnionFind(len(answers))
        for left in range(len(answers)):
            for right in range(left + 1, len(answers)):
                if self.equivalent(answers[left], answers[right]):
                    union_find.union(left, right)
        groups: dict[int, list[int]] = defaultdict(list)
        for index in range(len(answers)):
            groups[union_find.find(index)].append(index)
        clusters: list[SemanticCluster] = []
        total = len(answers)
        for indices in groups.values():
            canonical_counts: dict[str, int] = defaultdict(int)
            for index in indices:
                canonical_counts[answers[index].canonical] += 1
            representative = sorted(canonical_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
            lexical_consistency = max(canonical_counts.values()) / len(indices)
            member_ids = tuple(generation_ids[index] for index in indices)
            clusters.append(
                SemanticCluster(
                    cluster_id=stable_hash({"members": member_ids})[:20],
                    member_generation_ids=member_ids,
                    canonical_representative=representative,
                    mass=len(indices) / total,
                    lexical_consistency=lexical_consistency,
                )
            )
        return tuple(sorted(clusters, key=lambda cluster: (-cluster.mass, cluster.canonical_representative)))
