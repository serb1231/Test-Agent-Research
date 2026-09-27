# define the queries for every operation
append_database_insert_operation = """
            INSERT INTO traces (
                seq, op, agent, id_operation, kind_memory, type_memory, 
                importance_memory, created_at, derived_from, memory_text, qvec, embedding
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
                        created_at TIMESTAMP,
                        derived_from TEXT[],
                        memory_text TEXT,
                        embedding VECTOR(1536)
                    );
                    -- create index for the vector embeddings
                    CREATE INDEX hnsw_embedding_idx ON traces 
                    USING hnsw (embedding vector_cosine_ops) 
                    WHERE embedding IS NOT NULL;
                    
                    -- B-Tree index by agent and operation
                    CREATE INDEX btree_agent_op_idx ON traces (agent, op);
                    
                    -- B-Tree Index for Time-Series queries
                    CREATE INDEX btree_created_at_idx ON traces (created_at) WHERE created_at IS NOT NULL;
                """

