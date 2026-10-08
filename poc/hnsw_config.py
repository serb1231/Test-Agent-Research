"""HNSW build/search parameters, pinned once for both engines.

The point of this module is that the PG index DDL in PostgreSQL/queries.py and
the HnswConfigDiff in Qdrant/collections_config.py are no longer two places
holding the same numbers. m, ef_construction and ef_search are written here and
read by both, so "the same configuration" is a property of the code and not of
whoever last edited one of the two files.

Read the caveat at the bottom before concluding that equal numbers mean equal
searches -- they do not.
"""

# False reproduces the committed baseline (results-no-hnsw/): both planners pick
# their own access path, and both pick exact, because an `agent = ?` filter
# selects ~1% of the table in PG and ~5.7 MB of vectors in Qdrant.
# True removes that choice on both sides -- see force_hnsw_* notes below.
FORCE_HNSW = True

# ---------------------------------------------------------------- build params
# PG:     CREATE INDEX ... WITH (m = M, ef_construction = EF_CONSTRUCTION)
# Qdrant: HnswConfigDiff(m=M, ef_construct=EF_CONSTRUCTION)
M = 16
EF_CONSTRUCTION = 64

# --------------------------------------------------------------- search params
# PG:     SET hnsw.ef_search
# Qdrant: SearchParams(hnsw_ef=...)
#
# Must be >= k_prime (config.K_PRIME = 20) or the graph walk cannot return the
# requested number of candidates -- the silent-truncation trap in findings.md.
# 64 leaves headroom at k'=20; raise it with k', never below it.
EF_SEARCH = 64

# ------------------------------------------------- PG: how the graph is forced
# enable_sort = off is the whole trick. The btree plan needs a Sort node to put
# ~960 rows into distance order; the HNSW index scan supplies that order itself.
# Taking Sort away therefore leaves the graph as the only way to satisfy the
# ORDER BY, without touching the schema -- every index stays exactly as it is in
# the baseline run, so insertion cost is not silently changed along with the
# access path. (Verified on the 95,406-embedding table: the inner plan goes from
# `Index Scan using btree_agent_op_idx` + top-N heapsort to `Index Scan using
# hnsw_embedding_idx`.) The outer `ORDER BY similarity DESC` still sorts, on 20
# rows, which is why this is safe rather than a query rewrite.
#
# Dropping btree_agent_op_idx would force the graph too, but it also makes every
# INSERT cheaper than in the baseline, which would contaminate the insertion
# latency column.
PG_DISABLE_SORT = True

# pgvector cannot filter inside the index: one ef_search=64 pass returns the
# global top ~64 and then rechecks `agent` on the heap, which leaves ~0.6 of the
# 20 rows asked for. relaxed_order retries, widening the search, until k rows
# survive the filter; the outer ORDER BY re-sorts, so relaxed is free here and
# strict_order would only cost time. Confirmed to return a full 20/20 at k'=20.
PG_ITERATIVE_SCAN = "relaxed_order"

# Ceiling on tuples visited across those retries. The default 20,000 is already
# enough at this shape (~1% selectivity needs ~2,000 global candidates for 20
# surviving rows), but pinning it means a growing trace cannot quietly start
# returning short result sets.
PG_MAX_SCAN_TUPLES = 40000

# --------------------------------------------- Qdrant: how the graph is forced
# Two separate gates, and missing either one leaves Qdrant brute-forcing:
#
# 1. indexing_threshold (optimizers_config, in KB) -- a segment under it has no
#    HNSW graph at all, so searches over it are always exact no matter what the
#    search params say. 1 KB starts the optimizer on the first points. (0 would
#    not: in Qdrant 0 means "never index", the opposite of what is wanted here.)
# 2. full_scan_threshold (hnsw_config, in KB) -- the query planner. A filtered
#    subset smaller than this is brute-forced even when the graph exists. One
#    agent is ~960 points * 6 KB = ~5,760 KB, comfortably under the 10,000 KB
#    default, which is exactly why the baseline run came out exact.
#
# 10 is the floor Qdrant enforces on full_scan_threshold -- anything smaller is
# rejected at create_collection time with a 422, not silently clamped. All the
# value has to do is sit below the filtered subset, and 10 KB clears that by
# ~570x, so the floor costs nothing.
QDRANT_INDEXING_THRESHOLD_KB = 1
QDRANT_FULL_SCAN_THRESHOLD_KB = 10

# Belt and braces at query time: exact=False forbids the fallback outright, so a
# misconfigured threshold shows up as a worse recall number rather than as a
# silent exact search wearing an ANN label.
QDRANT_EXACT = False

# --------------------------------------------------------------- the caveat
# Equal m / ef_construct / ef_search does NOT make these the same search, and no
# setting in this file can:
#
#   * pgvector walks one global graph, applies `agent = ?` afterwards on the
#     heap, and retries until k rows survive.
#   * Qdrant traverses a filter-aware graph in a single pass.
#
# So this configuration equalises the knobs, not the algorithms. It is also an
# asymmetry in *when* the graph exists: PG maintains HNSW transactionally on
# every INSERT, so a forced search always walks a complete graph, while Qdrant's
# optimizer indexes asynchronously -- early searches in an interleaved
# insert/search trace can still hit an unindexed segment and be exact in spite of
# all of the above. verify_hnsw.py reports how much of the collection was indexed
# by the end of a run, which is the number to quote next to the latencies.
