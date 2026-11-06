"""Did the last run actually walk the graph?

Run this straight after PostgreSQL/main.py or Qdrant/main.py. Neither engine
raises when it quietly falls back to an exact search, so a forced-HNSW run and a
baseline run look identical in the logs and differ only in the latency and recall
numbers -- which is precisely the situation where you want evidence rather than a
config file that claims the right thing.

    python verify_hnsw.py            # both engines, whichever is reachable
    python verify_hnsw.py postgres
    python verify_hnsw.py qdrant
"""

import sys
from pathlib import Path

POC = Path(__file__).resolve().parent
sys.path.insert(0, str(POC))

# PostgreSQL/ and Qdrant/ both contain an `agent_logic` module, and both runners
# work because each is launched from inside its own folder. Importing from here
# means the two names collide, so each engine's modules are loaded with its own
# folder at the front of sys.path and the shared names evicted in between.
_SHARED_NAMES = ("agent_logic", "queries", "collections_config")


def engine_modules(folder, names):
    """Import `names` out of poc/<folder>/, isolated from the other engine."""
    for name in _SHARED_NAMES:
        sys.modules.pop(name, None)
    path = str(POC / folder)
    sys.path.insert(0, path)
    try:
        return [__import__(name) for name in names]
    finally:
        sys.path.remove(path)


from config import K_PRIME  # noqa: E402
from hnsw_config import (  # noqa: E402
    EF_CONSTRUCTION,
    EF_SEARCH,
    FORCE_HNSW,
    M,
)

PG_DSN = dict(host="localhost", dbname="postgres", user="postgres", password="password")


def check_postgres():
    import psycopg2

    agent_logic, queries = engine_modules("PostgreSQL", ("agent_logic", "queries"))
    session_settings = agent_logic.session_settings
    query_fetch_based_on_cosine = queries.query_fetch_based_on_cosine

    print("=" * 68)
    print("PostgreSQL")
    print("=" * 68)

    conn = psycopg2.connect(**PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    # 1. what the indexes were actually used for, over the whole run.
    # This is the headline: hnsw_embedding_idx at 0 scans means the run measured
    # an exact top-N sort no matter what hnsw.* was set to.
    cur.execute(
        """
        SELECT indexrelname, idx_scan
        FROM pg_stat_user_indexes
        WHERE relname = 'traces'
        ORDER BY idx_scan DESC
        """
    )
    rows = cur.fetchall()
    scans = dict(rows)
    print("\nindex scans since the last stats reset")
    for name, n in rows:
        print(f"  {name:<28} {n:>10,}")

    hnsw = scans.get("hnsw_embedding_idx", 0)
    btree = scans.get("btree_agent_op_idx", 0)
    searches = hnsw + btree
    if searches:
        print(f"\n  graph share of search path: {hnsw / searches:.1%} "
              f"({hnsw:,} hnsw / {btree:,} btree)")

    # 2. the build parameters as stored, not as intended
    cur.execute(
        """
        SELECT pg_get_indexdef(i.indexrelid)
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        WHERE c.relname = 'hnsw_embedding_idx'
        """
    )
    built = cur.fetchone()
    print(f"\nindex as built\n  {built[0] if built else 'MISSING'}")

    # 3. the plan for the real query, under the real session settings
    for statement in session_settings():
        cur.execute(statement)
    cur.execute("SELECT embedding FROM traces WHERE embedding IS NOT NULL LIMIT 1")
    probe = cur.fetchone()
    if probe is None:
        print("\n  table holds no embeddings -- run PostgreSQL/main.py first")
    else:
        vec = probe[0]
        cur.execute("SELECT agent FROM traces WHERE op = 'insert' LIMIT 1")
        agent = cur.fetchone()[0]
        cur.execute(
            "EXPLAIN (ANALYZE, COSTS off) " + query_fetch_based_on_cosine,
            (vec, agent, vec, K_PRIME),
        )
        plan = [line for (line,) in cur.fetchall()]
        print("\nplan for query_fetch_based_on_cosine under session_settings()")
        for line in plan:
            # the Order By line inlines the whole 1536-dim probe vector, ~19 kB
            # of float text that buries every other node in the plan
            print(f"  {line[:118]}{' ...' if len(line) > 118 else ''}")

        text = "\n".join(plan)
        walked = "hnsw_embedding_idx" in text
        print(f"\n  session GUCs: {' '.join(session_settings())}")
        verdict(walked, "pgvector HNSW index scan", "btree scan + exact top-N sort")

        # a forced graph that cannot fill k' is the silent-truncation trap
        cur.execute(query_fetch_based_on_cosine, (vec, agent, vec, K_PRIME))
        got = len(cur.fetchall())
        flag = "ok" if got == K_PRIME else "SHORT -- raise hnsw.ef_search / max_scan_tuples"
        print(f"  rows returned at k'={K_PRIME}: {got} ({flag})")

    cur.close()
    conn.close()


def check_qdrant():
    from qdrant_client import QdrantClient

    # collections_config only; Qdrant/agent_logic.py memory-maps the 600 MB
    # vectors file at import time and none of that is needed to read a plan.
    (collections_config,) = engine_modules("Qdrant", ("collections_config",))
    MEMORIES_COLLECTION = collections_config.MEMORIES_COLLECTION
    SEARCH_PARAMS = collections_config.SEARCH_PARAMS
    QDRANT_URL = "http://localhost:6333"

    print()
    print("=" * 68)
    print("Qdrant")
    print("=" * 68)

    client = QdrantClient(url=QDRANT_URL, timeout=30)
    info = client.get_collection(MEMORIES_COLLECTION)

    points = info.points_count or 0
    indexed = info.indexed_vectors_count or 0
    cfg = info.config.hnsw_config

    print(f"\nstatus                {info.status}")
    print(f"points                {points:,}")
    print(f"indexed vectors       {indexed:,}")
    # The one Qdrant-specific failure mode: the optimizer runs asynchronously, so
    # an interleaved insert/search trace can finish with part of the collection
    # still unindexed, and every search that touched those segments was exact.
    # indexed_vectors_count is summed per segment, and a segment waiting to be
    # merged can still hold vectors that another segment already indexed, so this
    # legitimately reads above 100% mid-optimization -- hence >= rather than ==
    # for the "fully indexed" test below.
    fully_indexed = points > 0 and indexed >= points
    if points:
        print(f"graph coverage        {min(indexed / points, 1.0):.1%}"
              f"{'  (raw ratio %.1f%%, segments mid-merge)' % (100 * indexed / points)
                 if indexed > points else ''}")
    print(f"\nhnsw_config as stored m={cfg.m} ef_construct={cfg.ef_construct} "
          f"full_scan_threshold={cfg.full_scan_threshold}")
    print(f"search params sent    {SEARCH_PARAMS}")

    # Vectors are 1536 * 4 B = 6 KB each; full_scan_threshold is in KB, and the
    # subset the filter selects has to clear it or the planner brute-forces.
    per_agent_kb = (points / 100) * 6 if points else 0
    print(f"\nfiltered subset       ~{per_agent_kb:,.0f} KB per agent "
          f"vs full_scan_threshold {cfg.full_scan_threshold} KB")

    walked = (
        fully_indexed
        and cfg.full_scan_threshold is not None
        and per_agent_kb > cfg.full_scan_threshold
    )
    verdict(
        walked,
        "filter-aware graph traversal",
        "brute force over the filtered subset",
        basis="config implies",
    )
    if points and not fully_indexed:
        print("  (coverage below 100% means some searches in the run were exact "
              "regardless of the thresholds -- quote this next to the latencies)")

    client.close()


def verdict(walked, yes, no, basis="observed"):
    """`basis` keeps the two engines' evidence honestly labelled.

    PG's answer comes from an EXPLAIN ANALYZE of the real query, so it is
    observed. Qdrant exposes no per-query plan, so its answer is inferred from
    the stored thresholds and the indexed-vector count -- strong evidence, but
    not the same thing, and the distinction belongs in the output rather than in
    whoever reads it later.
    """
    got = yes if walked else no
    want = yes if FORCE_HNSW else no
    mark = "OK  " if walked == FORCE_HNSW else "MISMATCH"
    print(f"  [{mark}] FORCE_HNSW={FORCE_HNSW} wants {want}; {basis} {got}")


if __name__ == "__main__":
    print(f"shared config: FORCE_HNSW={FORCE_HNSW} m={M} "
          f"ef_construction={EF_CONSTRUCTION} ef_search={EF_SEARCH} k'={K_PRIME}")
    which = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    for name, fn in (("postgres", check_postgres), ("qdrant", check_qdrant)):
        if which not in (name, "both"):
            continue
        try:
            fn()
        except Exception as exc:
            # one unreachable engine must not hide the other's report
            print(f"\n{name}: could not check ({type(exc).__name__}: {exc})")
