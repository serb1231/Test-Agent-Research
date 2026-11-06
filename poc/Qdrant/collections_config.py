import sys
from pathlib import Path

from qdrant_client import models

# hnsw_config lives in poc/, one level up, and is the single source of truth for
# the parameters this file and ../PostgreSQL/queries.py share.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hnsw_config import (  # noqa: E402
    EF_CONSTRUCTION,
    EF_SEARCH,
    FORCE_HNSW,
    M,
    QDRANT_EXACT,
    QDRANT_FULL_SCAN_THRESHOLD_KB,
    QDRANT_INDEXING_THRESHOLD_KB,
)

# Mirrors the PostgreSQL schema in ../PostgreSQL/queries.py.
#
# PostgreSQL keeps every trace row (insert / search / touch) in one `traces`
# table because a NULL embedding is free there. Qdrant wants a vector per
# point, so the trace is split in two:
#   MEMORIES_COLLECTION  -> the insert ops, the only ones that are searchable
#   OPS_COLLECTION       -> the search / touch audit rows, payload only
# The split keeps the log-write cost measurable, the same as the PG version.

MEMORIES_COLLECTION = "traces"
OPS_COLLECTION = "trace_ops"

VECTOR_SIZE = 1536

# Build parameters come from ../hnsw_config.py, the same module the PG DDL reads,
# so `m` and `ef_construct` here cannot drift from `m` and `ef_construction` there.
#
# full_scan_threshold is Qdrant's query planner, measured in KB of vectors: a
# filtered subset smaller than the threshold is brute-forced, a larger one goes
# through the graph. One agent is ~960 points * 6 KB = ~5,760 KB, which sits under
# the 10,000 KB default -- that is the entire reason the baseline run came back
# exact (results-no-hnsw/), matching PG, whose planner reached the same verdict
# from the other direction. Dropping the threshold to Qdrant's minimum of 10 KB
# puts every filtered subset ~570x above the line, so the graph is always on the
# hot path.
HNSW_CONFIG = models.HnswConfigDiff(
    m=M,
    ef_construct=EF_CONSTRUCTION,
    **(
        {"full_scan_threshold": QDRANT_FULL_SCAN_THRESHOLD_KB}
        if FORCE_HNSW
        else {}
    ),
)

# The second gate, and the one that is easy to miss: a segment holding less than
# indexing_threshold KB has no HNSW graph built for it at all, so searches over it
# are exact however full_scan_threshold is set. At the 10,000 KB default the graph
# does get built eventually -- the collection is ~586 MB -- but on Qdrant's own
# schedule, which in an interleaved insert/search trace means a long opening
# stretch of brute-force searches. 1 KB starts the optimizer on the first points.
#
# None means "send no optimizers_config at all", i.e. keep Qdrant's defaults.
OPTIMIZERS_CONFIG = (
    models.OptimizersConfigDiff(indexing_threshold=QDRANT_INDEXING_THRESHOLD_KB)
    if FORCE_HNSW
    else None
)

# hnsw_ef is the analogue of pgvector's hnsw.ef_search and is pinned to the same
# EF_SEARCH. exact=False is belt and braces: it forbids the exact fallback at
# query time, so if a threshold above is ever misconfigured the result is a worse
# recall number rather than a silent brute-force search wearing an ANN label.
#
# Note this is a genuine handicap relative to letting Qdrant choose: measured on a
# 24k-point replica of this workload, hnsw_ef=40 scored recall@20 0.9930 where
# Qdrant's own default scored 0.9995. That is the price of a graph-to-graph
# comparison, not a bug -- and the two filtered-ANN algorithms still are not
# equivalent (see the caveat at the foot of ../hnsw_config.py).
SEARCH_PARAMS = (
    models.SearchParams(hnsw_ef=EF_SEARCH, exact=QDRANT_EXACT) if FORCE_HNSW else None
)

VECTORS_CONFIG = models.VectorParams(
    size=VECTOR_SIZE,
    distance=models.Distance.COSINE,
)


def memory_payload(item):
    """One insert op, keyed like the columns of the PG `traces` table."""
    return {
        "seq": item["seq"],
        "op": item["op"],
        "agent": item["agent"],
        "id_operation": item["id"],
        "kind_memory": item["kind"],
        "type_memory": item["type"],
        "importance_memory": item["importance"],
        "sim_t": item["sim_t"],
        "last_accessed_at": item["sim_t"],  # same as created for a new memory
        "derived_from": item.get("derived_from", []),
        "memory_text": item["text"],
        "qvec": item.get("vec"),
    }


def search_payload(item):
    return {
        "seq": item["seq"],
        "op": item["op"],
        "agent": item["agent"],
        "kind_memory": item["kind"],
        "trigger": item.get("trigger"),
        "filter": item.get("filter"),
        "qvec": item.get("qvec"),
        "k": item.get("k"),
        "k_prime": item.get("k_prime"),
        "sim_time": item.get("sim_t"),
        "pool": item.get("pool"),
        "composite_top": item.get("composite_top"),
    }


def touch_payload(item):
    return {
        "seq": item["seq"],
        "op": item["op"],
        "agent": item["agent"],
        "kind_memory": item["kind"],
        "ids": item.get("ids", []),
        "at": item.get("at", item["sim_t"]),
    }


def agent_filter(agent):
    """Equivalent of `WHERE agent = %s` on the memories collection."""
    return models.Filter(
        must=[models.FieldCondition(key="agent", match=models.MatchValue(value=agent))]
    )


def ids_filter(id_operations):
    """Equivalent of `WHERE id_operation IN %s`."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="id_operation", match=models.MatchAny(any=list(id_operations))
            )
        ]
    )


def create_collections(client):
    """Drop and rebuild both collections, like `database_creation` does."""
    # a bare delete_collection on a missing collection just returns false, but
    # guarding it keeps a fresh server from looking like a failed drop in the log
    for collection in (MEMORIES_COLLECTION, OPS_COLLECTION):
        if client.collection_exists(collection):
            client.delete_collection(collection)

    create_kwargs = {}
    if OPTIMIZERS_CONFIG is not None:
        create_kwargs["optimizers_config"] = OPTIMIZERS_CONFIG
    client.create_collection(
        collection_name=MEMORIES_COLLECTION,
        vectors_config=VECTORS_CONFIG,
        hnsw_config=HNSW_CONFIG,
        **create_kwargs,
    )
    # the two payload indexes that back agent_filter() and ids_filter(),
    # standing in for the PG btree on (agent, op) and the UNIQUE id_operation
    client.create_payload_index(
        collection_name=MEMORIES_COLLECTION,
        field_name="agent",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=MEMORIES_COLLECTION,
        field_name="id_operation",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    # vectors_config={} -> payload-only collection, nothing to search here
    client.create_collection(collection_name=OPS_COLLECTION, vectors_config={})
