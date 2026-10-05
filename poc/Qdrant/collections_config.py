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

# Build parameters match the PG index, and that is the whole of what is pinned:
#   CREATE INDEX ... USING hnsw (...) WITH (m = 16, ef_construction = 64)
#
# Everything else is left at Qdrant's defaults deliberately. The point is to let
# each engine pick its own access path, because that is what the PG side does.
#
# full_scan_threshold is Qdrant's query planner: a filtered subset smaller than
# the threshold is brute-forced, a larger one goes through the graph. It used to
# be pinned to 10 KB here to keep the graph on the hot path, on the theory that
# the PG side was using its HNSW index. It was not -- across a full run
# pg_stat_user_indexes reports 58,443 scans of btree_agent_op_idx and 0 of
# hnsw_embedding_idx, because `agent = ?` selects ~1% of the table and PG's
# planner costs a btree scan plus an exact top-N sort as cheaper than an index
# pgvector cannot filter inside. Qdrant's default (10000 KB) reaches the same
# verdict on the same data: ~960 vectors * 6 KB = ~5760 KB is under it, so the
# filtered subset is scanned exactly. Both planners now choose, and both choose
# exact, so the quality numbers measure the engine instead of this override.
HNSW_CONFIG = models.HnswConfigDiff(m=16, ef_construct=64)

# Same reasoning one level up. indexing_threshold was pinned to 1 KB to start the
# optimizer immediately; at its default (10000 KB) the graph is still built, since
# the collection is ~586 MB, just on Qdrant's own schedule. None here means "send
# no optimizers_config at all".
OPTIMIZERS_CONFIG = None

# No search params either. Pinning hnsw_ef=40 to mirror pgvector's ef_search
# default only equalises anything if PG actually walks its graph, and it does not.
# Measured on a 24k-point replica of this workload, hnsw_ef=40 scored recall@20
# 0.9930 where Qdrant's own default scored 0.9995 -- so the pin was a handicap,
# not a control.
#
# Worth knowing if these are ever compared graph-to-graph: the two filtered-ANN
# algorithms cannot be made equivalent. pgvector searches a global graph, applies
# the filter afterwards on the heap, and (with hnsw.iterative_scan) retries until
# k rows survive; Qdrant traverses a filter-aware graph in one pass. Equal m and
# ef_construct does not make those the same search.
SEARCH_PARAMS = None

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
