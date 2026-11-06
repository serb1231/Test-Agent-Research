"""HNSW build/search parameters

m, ef_construction and ef_search are written here and read by both PostgreSQL and Qdrant,
in order for them to have the same configuration
"""

# force the databases to use HNSW
FORCE_HNSW = True

# BUILD PARAMS
# PG:     CREATE INDEX ... WITH (m = M, ef_construction = EF_CONSTRUCTION)
# Qdrant: HnswConfigDiff(m=M, ef_construct=EF_CONSTRUCTION)
M = 16
EF_CONSTRUCTION = 64

# SEARCH PARAMS
# PG:     SET hnsw.ef_search
# Qdrant: SearchParams(hnsw_ef=...)
#
# Must be >= k_prime (config.K_PRIME = 20)
EF_SEARCH = 64


PG_DISABLE_SORT = True

# In order for PG to return the correct nr of entries for hnsw (otherwise, the
# parsing based on agent id will trim it down to 1-2)
PG_ITERATIVE_SCAN = "relaxed_order"

# Ceiling on tuples visited across those retries.
PG_MAX_SCAN_TUPLES = 40000

# QDRANT
QDRANT_INDEXING_THRESHOLD_KB = 1
QDRANT_FULL_SCAN_THRESHOLD_KB = 10

# Belt and braces at query time: exact=False forbids the fallback outright
QDRANT_EXACT = False


#   * pgvector walks one global graph, applies `agent = ?` afterwards on the
#     heap, and retries until k rows survive.
#   * Qdrant traverses a filter-aware graph in a single pass.