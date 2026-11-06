import sys
from pathlib import Path

# hnsw_config lives in poc/, one level up, and is the single source of truth for
# the build parameters this DDL and ../Qdrant/collections_config.py share.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hnsw_config import EF_CONSTRUCTION, M  # noqa: E402

append_database_insert_operation = """
    INSERT INTO traces (
        seq, op, agent, id_operation, kind_memory, type_memory, 
        importance_memory, sim_t, last_accessed_at, derived_from, memory_text, qvec, embedding
    ) VALUES %s
"""

append_database_search_operation = """
    INSERT INTO traces (
        seq, op, agent, kind_memory, trigger, filter, 
        qvec, k, k_prime, sim_time, pool, composite_top
    ) VALUES %s
"""

append_database_touch_operation = """
    INSERT INTO traces (
        seq, op, agent, kind_memory, ids, at
    ) VALUES %s
"""

database_creation = """
    DROP TABLE IF EXISTS traces CASCADE;

    CREATE TABLE traces (
        seq INTEGER PRIMARY KEY,
        op VARCHAR(50),
        agent VARCHAR(50),
        id_operation VARCHAR(100) UNIQUE,
        kind_memory VARCHAR(50),
        trigger TEXT,
        filter JSONB NULL,
        qvec INTEGER NULL,
        k INTEGER NULL,
        k_prime INTEGER NULL,
        sim_time TIMESTAMP NULL,
        pool JSONB NULL,
        composite_top JSONB NULL,
        ids TEXT[] NULL,
        at TIMESTAMP NULL,
        type_memory INTEGER,
        importance_memory INTEGER,
        sim_t TIMESTAMP,
        last_accessed_at TIMESTAMP,
        derived_from TEXT[],
        memory_text TEXT,
        embedding VECTOR(1536)
    );

    CREATE INDEX hnsw_embedding_idx ON traces 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {m}, ef_construction = {ef_construction})
    WHERE embedding IS NOT NULL;

    CREATE INDEX btree_agent_op_idx ON traces (agent, op);

    CREATE INDEX btree_sim_t_idx ON traces (sim_t) WHERE sim_t IS NOT NULL;
""".format(m=M, ef_construction=EF_CONSTRUCTION)

query_fetch_based_on_cosine = """
WITH r as MATERIALIZED(
    SELECT id_operation, memory_text, 1 - (embedding <=> %s::vector) AS similarity
    FROM traces
    WHERE embedding is NOT NULL AND op = 'insert' AND agent = %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s)
select * from r order by similarity DESC;
"""