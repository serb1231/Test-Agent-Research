# define the queries for every operation
query_insert = """
            INSERT INTO traces (
                seq, op, agent, id_operation, kind_memory, type_memory, 
                importance_memory, created_at, derived_from, memory_text, qvec, embedding
            ) VALUES %s
        """

query_search = """
            INSERT INTO traces (
                seq, op, agent, kind_memory, trigger, filter, 
                qvec, k, k_prime, sim_time, pool, composite_top
            ) VALUES %s
        """

query_touch = """
            INSERT INTO traces (
                seq, op, agent, kind_memory, ids, at
            ) VALUES %s
        """

database_creation = """
                    DROP TABLE IF EXISTS traces CASCADE;
                    CREATE TABLE traces (
                        seq SERIAL PRIMARY KEY,
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
                """