"""The store-agnostic interface every backend implements.

Mirrors the adapter protocol from the run plan (§05): the simulator and
retrieval/rescoring logic only ever talk to this, never to Chroma (or,
later, pgvector/Qdrant) directly. Swapping backends means writing a new
class here, nothing upstream changes.
"""

from typing import Protocol, Optional
from datetime import datetime


class MemoryStore(Protocol):
    def insert(self, id: str, vec: list[float], meta: dict) -> None:
        """meta carries agent_id, type, text, importance, created_at, last_accessed_at."""
        ...

    def search(
        self,
        qvec: list[float],
        filter: dict,
        k_prime: int,
        now: datetime,
    ) -> list[dict]:
        """Return up to k_prime candidates as dicts with id/meta/distance.

        `now` is sim time, passed explicitly (never wall clock) so recency
        decay is computed against the simulation's clock.
        """
        ...

    def touch(self, ids: list[str], at: datetime) -> None:
        """Bump last_accessed_at on the given ids without touching the vector."""
        ...

    def update(self, id: str, fields: dict) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def stats(self) -> dict:
        """Backend-specific counters (row/point count at minimum)."""
        ...
