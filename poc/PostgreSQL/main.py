import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import pandas as pd
from queries import *
import typing
from typing import List

import json

class TypeOfDataNotFound(Exception):
    pass

def list_for_insertion(item, vector):
    return (
        item["seq"],
        item["op"],
        item["agent"],
        item["id"],
        item["kind"],
        item["type"],
        item["importance"],
        item["created_at"],
        item.get("derived_from", []),
        item["text"],
        item.get("vec")
    )

def list_for_search(item):
    return (
        item["seq"],
        item["op"],
        item["agent"],
        item["kind"],
        item.get("trigger"),
        json.dumps(item["filter"]) if "filter" in item else None,
        item.get("qvec"),
        item.get("k"),
        item.get("k_prime"),
        item.get("sim_t"),
        json.dumps(item["pool"]) if "pool" in item else None,
        json.dumps(item["composite_top"]) if "composite_top" in item else None
    )

def list_for_touch(item):
    return (
        item["seq"],
        item["op"],
        item["agent"],
        item["kind"],
        item.get("ids", []),
        item.get("at")
    )

def read_trace_data() -> typing.Tuple[List, List]:
    # read and parse the json file
    data_seed = []
    data_generated = []
    with open("../trace.jsonl", "r", encoding="utf-8") as f:
        seed_data = True
        for i, line in enumerate(f):
            if line.strip():  # Skip empty lines
                if seed_data:
                    data_seed.append(json.loads(line))
                    if data_seed[-1]["kind"] != "seed":
                        data_generated.append(data_seed.pop(-1))
                        seed_data = False
                else:
                    data_generated.append(json.loads(line))

    print(f"Loaded {len(data_seed)} records.")
    print(f"Loaded {len(data_generated)} records.")
    return data_seed, data_generated

def initial_setup_agents_seed():
    with psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password") as conn:
        with conn.cursor() as cursor:
            cursor.execute("""CREATE EXTENSION IF NOT EXISTS vector;""")

            cursor.execute(database_creation)

            # create vector index
            cursor.execute("""CREATE INDEX IF NOT EXISTS traces_embedding_idx ON traces USING HNSW (embedding vector_cosine_ops);""")


            data_seed, _ = read_trace_data()

            # Route the data into distinct lists
            tuples_insert = []
            tuples_search = []
            tuples_touch = []

            for item in data_seed:
                op = item.get("op")

                if op == "insert":
                    tuples_insert.append(list_for_insertion(item))

                elif op == "search":
                    tuples_search.append(list_for_search(item))

                elif op == "touch":
                    tuples_touch.append(list_for_touch(item))
                else:
                    raise TypeOfDataNotFound(f"Unknown operation type: {op}")

            if tuples_insert:
                execute_values(cursor, query_insert, tuples_insert)


            if tuples_search:
                execute_values(cursor, query_search, tuples_search)

            if tuples_touch:
                execute_values(cursor, query_touch, tuples_touch)

            conn.commit()

            print("Successfully put the data inside")