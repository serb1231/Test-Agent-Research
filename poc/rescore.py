"""Composite scoring — the article's central point (§00): a vector index
answers 'what's geometrically closest', not 'what should the agent recall'.
This function is store-agnostic on purpose: it takes whatever candidates
ChromaStore.search() returned and ranks them the same way regardless of
which backend produced them, so swapping backends later never changes
this logic.
"""

import math
from datetime import datetime

from config import ALPHA, BETA, GAMMA, RECENCY_DECAY


def composite_score(candidate: dict, now: datetime) -> float:
    meta = candidate["meta"]
    last_accessed_at = meta["last_accessed_at"]
    if isinstance(last_accessed_at, (int, float)):
        last_accessed_dt = datetime.fromtimestamp(last_accessed_at)
    else:
        last_accessed_dt = last_accessed_at
    hours_elapsed = max((now - last_accessed_dt).total_seconds() / 3600.0, 0.0)
    recency = RECENCY_DECAY ** hours_elapsed  # decays from last retrieval, not creation — a touch resets this

    importance = meta.get("importance", 0) / 10.0

    # Chroma cosine distance is 1 - cosine_similarity; convert back.
    distance = candidate.get("distance")
    #search returns distances as 0.5,0.1 where less is more similar , we convert to relevance as 1-distance so that more is better
    relevance = 1.0 - distance if distance is not None else 0.0

    return ALPHA * recency + BETA * importance + GAMMA * relevance


#rescore and return the top k candidates after rescoring , k_prime is the number of candidates returned from the store before rescoring
def rescore(candidates: list[dict], now: datetime, k: int) -> list[dict]:
    scored = [(composite_score(c, now), c) for c in candidates]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:k]]
