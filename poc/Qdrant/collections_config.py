from qdrant_client import models

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

# same HNSW build parameters as the PG index, so the two are comparable:
#   CREATE INDEX ... USING hnsw (...) WITH (m = 16, ef_construction = 64)
#
# full_scan_threshold matters as much as m/ef_construct here. Qdrant decides per
# query whether to walk the graph or to brute-force the filtered subset, and the
# default threshold is 10000 KB. Every search in the trace is filtered down to a
# single agent, which is ~954 vectors * 6 KB = ~5700 KB, i.e. under the default.
# Left alone, every search would run exact and the graph would never be touched,
# so the m/ef_construct numbers above would be measuring nothing. 10 KB is the
# lowest the server accepts, and it keeps the graph on the hot path, which is what
# the PG side builds its index for.
HNSW_CONFIG = models.HnswConfigDiff(m=16, ef_construct=64, full_scan_threshold=10)

# Same story one level up: a segment only gets an HNSW index once its vector data
# passes indexing_threshold (default 20000 KB), and until then search is a plain
# scan. The run inserts and searches interleaved, so the default would have the
# early searches exact and the later ones indexed -- a latency distribution that
# is a blend of two algorithms and comparable to nothing. 1 KB starts the
# optimizer immediately. (0 would mean "never index", not "always".)
OPTIMIZERS_CONFIG = models.OptimizersConfigDiff(indexing_threshold=1)

# pgvector's hnsw.ef_search defaults to 40 and the PG side never overrides it, so
# the Qdrant searches are pinned to the same beam width instead of its default.
SEARCH_PARAMS = models.SearchParams(hnsw_ef=40)

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

    client.create_collection(
        collection_name=MEMORIES_COLLECTION,
        vectors_config=VECTORS_CONFIG,
        hnsw_config=HNSW_CONFIG,
        optimizers_config=OPTIMIZERS_CONFIG,
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
