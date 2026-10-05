append_database_insert_operation = """
    EXPLAIN (ANALYZE, BUFFERS)
    INSERT INTO traces (
        seq, op, agent, id_operation, kind_memory, type_memory, 
        importance_memory, sim_t, last_accessed_at, derived_from, memory_text, qvec, embedding
    ) VALUES %s
"""

append_database_search_operation = """
    EXPLAIN (ANALYZE, BUFFERS)
    INSERT INTO traces (
        seq, op, agent, kind_memory, trigger, filter, 
        qvec, k, k_prime, sim_time, pool, composite_top
    ) VALUES %s
"""

append_database_touch_operation = """
EXPLAIN (ANALYZE, BUFFERS)
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
    WITH (m = 16, ef_construction = 64)
    WHERE embedding IS NOT NULL;

    CREATE INDEX btree_agent_op_idx ON traces (agent, op);

    CREATE INDEX btree_sim_t_idx ON traces (sim_t) WHERE sim_t IS NOT NULL;
"""

query_fetch_based_on_cosine = """
WITH r as MATERIALIZED(
    EXPLAIN SELECT id_operation, memory_text, 1 - (embedding <=> %s::vector) AS similarity
    FROM traces
    WHERE embedding is NOT NULL AND op = 'insert' AND agent = %s
    ORDER BY embedding <=> %s::vector
    LIMIT %s)
EXPLAIN select * from r order by similarity DESC;
"""