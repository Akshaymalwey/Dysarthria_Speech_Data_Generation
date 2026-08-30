"""
Samples which disfluency category (or categories) each sentence should
receive, before the LLM ever sees the sentence -- matching the paper's
description: "prompted with the sentence to transform along with the target
disfluency category or categories for that sentence, sampled beforehand."
"""

import random
from typing import List

from config import (
    DISFLUENCY_TYPE_WEIGHTS,
    MIN_TYPES_PER_SENTENCE,
    MAX_TYPES_PER_SENTENCE,
)


def sample_disfluency_types(rng: random.Random = None) -> List[str]:
    """Return a non-empty list of 1-3 disfluency category names."""
    rng = rng or random
    types = list(DISFLUENCY_TYPE_WEIGHTS.keys())
    weights = list(DISFLUENCY_TYPE_WEIGHTS.values())

    n = rng.randint(MIN_TYPES_PER_SENTENCE, MAX_TYPES_PER_SENTENCE)
    n = min(n, len(types))

    # Weighted sample without replacement.
    chosen = []
    pool = list(zip(types, weights))
    for _ in range(n):
        total = sum(w for _, w in pool)
        r = rng.uniform(0, total)
        upto = 0
        for i, (t, w) in enumerate(pool):
            upto += w
            if upto >= r:
                chosen.append(t)
                pool.pop(i)
                break
    return chosen


def assign_labels(sentences: List[str], seed: int = 42):
    """Given a list of sentences, return list of (sentence, [types]) tuples."""
    rng = random.Random(seed)
    return [(s, sample_disfluency_types(rng)) for s in sentences]
