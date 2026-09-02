"""Chroma implementation of MemoryStore.

One collection, cosine distance, everything that isn't the vector goes into
Chroma's per-point metadata dict. Chroma metadata values must be
str/int/float/bool, so list-valued fields (derived_from) are JSON-encoded.

Composite rescoring (recency/importance/relevance) is NOT done here — this
adapter only returns raw ANN candidates plus their metadata. Rescoring lives
in sim.py so it's identical regardless of backend, matching the run plan's
"rescoring can't be avoided, and where it runs is architectural" point.
"""

import json
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings


def _to_epoch(dt: datetime) -> float:
    return dt.timestamp()


def _meta_out(id: str, meta: dict, document: str, distance: float | None) -> dict:
    m = dict(meta)
    if "derived_from" in m and isinstance(m["derived_from"], str):
        try:
            m["derived_from"] = json.loads(m["derived_from"])
        except (json.JSONDecodeError, TypeError):
            m["derived_from"] = []
    return {
        "id": id,
        "text": document,
        "meta": m,
        "distance": distance,
    }


class ChromaStore:
    def __init__(self, path: str, collection_name: str = "memories"):
        self.client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def insert(self, id: str, vec: list[float], meta: dict) -> None:
        m = dict(meta)
        if "derived_from" in m:
            m["derived_from"] = json.dumps(m.get("derived_from") or [])
        if isinstance(m.get("created_at"), datetime):
            m["created_at"] = _to_epoch(m["created_at"])
        if isinstance(m.get("last_accessed_at"), datetime):
            m["last_accessed_at"] = _to_epoch(m["last_accessed_at"])
        text = m.pop("text", "")
        self.collection.add(ids=[id], embeddings=[vec], metadatas=[m], documents=[text])

    def search(self, qvec: list[float], filter: dict, k_prime: int, now: datetime) -> list[dict]:
        where = filter if filter else None
        n = min(k_prime, max(self.collection.count(), 1))
        res = self.collection.query(
            query_embeddings=[qvec],
            n_results=n,
            where=where,
            include=["metadatas", "distances", "documents"],
        )
        if not res["ids"] or not res["ids"][0]:
            return []
        out = []
        for id_, meta, doc, dist in zip(
            res["ids"][0], res["metadatas"][0], res["documents"][0], res["distances"][0]
        ):
            out.append(_meta_out(id_, meta, doc, dist))
        return out

    def touch(self, ids: list[str], at: datetime) -> None:
        if not ids:
            return
        existing = self.collection.get(ids=ids, include=["metadatas"])
        new_metas = []
        for m in existing["metadatas"]:
            m = dict(m)
            m["last_accessed_at"] = _to_epoch(at)
            new_metas.append(m)
        self.collection.update(ids=existing["ids"], metadatas=new_metas)

    def update(self, id: str, fields: dict) -> None:
        existing = self.collection.get(ids=[id], include=["metadatas"])
        if not existing["ids"]:
            return
        m = dict(existing["metadatas"][0])
        for k, v in fields.items():
            if isinstance(v, datetime):
                v = _to_epoch(v)
            if k == "derived_from":
                v = json.dumps(v or [])
            m[k] = v
        self.collection.update(ids=[id], metadatas=[m])

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def stats(self) -> dict:
        return {"count": self.collection.count()}

    def get(self, id: str) -> dict | None:
        res = self.collection.get(ids=[id], include=["metadatas", "documents"])
        if not res["ids"]:
            return None
        return _meta_out(res["ids"][0], res["metadatas"][0], res["documents"][0], None)
