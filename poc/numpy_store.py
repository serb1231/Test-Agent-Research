"""The golden/reference store — exact brute-force cosine search over an
in-memory dict. Same MemoryStore shape as ChromaStore, but nothing here is
approximate: whatever it returns for a query IS the ground truth, by
construction. This is what the article's §01 calls running the simulation
"once, against a reference store that is neither Postgres nor Qdrant."
"""

import numpy as np


def _cosine_distance(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


class NumpyReference:
    def __init__(self):
        self._rows: dict[str, dict] = {}

    def insert(self, id: str, vec, meta: dict) -> None:
        m = dict(meta)
        text = m.pop("text", "")
        self._rows[id] = {"vec": vec, "meta": m, "text": text}

    def search(self, qvec, filter: dict, k_prime: int, now) -> list[dict]:
        candidates = []
        for id_, row in self._rows.items():
            if filter and not all(row["meta"].get(k) == v for k, v in filter.items()):
                continue
            dist = _cosine_distance(qvec, row["vec"])
            candidates.append({"id": id_, "text": row["text"], "meta": dict(row["meta"]), "distance": dist})
        candidates.sort(key=lambda c: c["distance"])
        return candidates[:k_prime]

    def touch(self, ids: list[str], at) -> None:
        for id_ in ids:
            if id_ in self._rows:
                self._rows[id_]["meta"]["last_accessed_at"] = at

    def update(self, id: str, fields: dict) -> None:
        if id in self._rows:
            self._rows[id]["meta"].update(fields)

    def delete(self, ids: list[str]) -> None:
        for id_ in ids:
            self._rows.pop(id_, None)

    def stats(self) -> dict:
        return {"count": len(self._rows)}

    def get(self, id: str):
        row = self._rows.get(id)
        if not row:
            return None
        return {"id": id, "text": row["text"], "meta": dict(row["meta"]), "distance": None}
