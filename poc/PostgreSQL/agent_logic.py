import logging
import time
from pathlib import Path
from typing import Any

import psycopg2
import threading
import json
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector

from queries import *
import numpy as np
import numpy.typing as npt
import math

current_folder = Path(__file__).resolve().parent
parent_folder = current_folder.parent

VECTORS_PATH = parent_folder / "data/vectors.npy"

def list_for_insertion(item, vector):
    return (
        item["seq"],
        item["op"],
        item["agent"],
        item["id"],
        item["kind"],
        item["type"],
        item["importance"],
        item["sim_t"],
        item["sim_t"], # last_accessed_at is the same as created for a new memory
        item.get("derived_from", []),
        item["text"],
        item.get("vec"),
        vector
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
        item.get("at", item["sim_t"])
    )


# NDGC (Normalised Discounted Cummulative Gain
def ndgc_compute(entries: list[int], g_t: list[int]):
    dgc = 0
    idcg = 0
    logging.debug(f"entries: {entries};\n ground truth:{g_t}")
    for i, entry in enumerate(entries):
        # find the entry position inside the g_t
        dgc += (20.0 / (1 + g_t.index(entry))) / math.log(i + 2, 2) if entry in g_t else 0
        idcg += (20.0 / (1 + i)) / math.log(i + 2, 2)
    logging.debug(f"ndgc: {dgc / idcg}")
    return dgc / idcg

def overlap(entries, g_t):
    sequences_retrieved = set(entry["id"] for entry in entries)
    sequences_ground_truth = set(entry["id"] for entry in g_t["pool"])

    # Calculate True Positives (overlap)
    true_positives = len(sequences_retrieved.intersection(sequences_ground_truth))

    # How many entries are in both lists
    overlap_ration = true_positives / len(sequences_retrieved) if sequences_retrieved else 0.0

    return overlap_ration

# Precision @5
def precision_5(entries, g_t):
    top_5_retrieved = set(entry["id"] for entry in entries[:5])
    top_5_ground_truth = set(entry["id"] for entry in g_t["pool"][:5])
    correct_entries = len(top_5_retrieved.intersection(top_5_ground_truth))
    precision_top_5 = correct_entries / 5.0

    return precision_top_5

class Agent:
    # data generated is only the one that belongs to this agent
    def __init__(self, id_agent: int, data_generated):
        self.cursor = None
        self.connection = None
        self.id_agent = id_agent
        self.entries_list = data_generated
        logging.info(f"Agent {id_agent}: initialized with {len(self.entries_list)} operations to process.")
        # self.cum_insertion_time = 0
        # self.cum_modif_time = 0
        # self.cum_search_time = 0
        self.insertion_times: list[int] = []
        self.log_times: list[int] = []
        self.modification_times: list[int] = []
        self.search_times: list[int] = []

        # metrics for searching database
        self.total_searches = 0
        self.cum_overlap = 0.0  # Replacing the redundant precision/recall
        self.cum_ndcg = 0.0
        self.cum_p_at_5 = 0.0
        self.cum_mrr = 0.0

    def run(self):
        vectors : npt.NDArray[np.float32] = np.load(VECTORS_PATH, mmap_mode='r')
        logging.info(f"Agent {self.id_agent}: started processing operations.")
        # start the connection in the run methods
        self.connection = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password")
        self.cursor = self.connection.cursor()
        seq = None
        try:
            self.connection.autocommit = True
            register_vector(self.connection)
            with self.cursor:
                # setting this will do good to the combination of vector cosine and B-Tree
                self.cursor.execute("SET hnsw.iterative_scan = relaxed_order;")
                for memory in self.entries_list:
                    seq = memory.get('seq')
                    if memory["op"] == "insert":
                        logging.debug(f"Executing INSERT for seq {seq}")
                        # add them directly to the database
                        insrt = list_for_insertion(memory, vectors[int(memory["vec"])].tolist())
                        start = time.perf_counter_ns()
                        execute_values(self.cursor, append_database_insert_operation, [insrt])
                        self.insertion_times.append(time.perf_counter_ns() - start)

                    elif memory["op"] == "search":
                        logging.debug(f"Executing SEARCH for seq {seq}")
                        insrt = list_for_search(memory)
                        start = time.perf_counter_ns()
                        execute_values(self.cursor, append_database_search_operation, [insrt])
                        self.log_times.append(time.perf_counter_ns() - start)
                        # get the vector of the trigger
                        embedding_of_trigger = vectors[int(memory["qvec"])].tolist()

                        start = time.perf_counter_ns()
                        self.cursor.execute(query_fetch_based_on_cosine, (embedding_of_trigger, memory["agent"], embedding_of_trigger, int(memory["k_prime"])))
                        self.search_times.append(time.perf_counter_ns() - start)

                        raw_results: list[tuple[Any, ...]] = self.cursor.fetchall()
                        formatted_results = []
                        for row in raw_results:
                            formatted_results.append({
                                "id" : row[0],
                                "text": row[1],
                                "cosine": row[2]
                                })

                        overlap_ration = overlap(formatted_results, memory)
                        self.cum_overlap += overlap_ration

                        precision_top_5 = precision_5(formatted_results, memory)
                        self.cum_p_at_5 += precision_top_5


                        ndcg = ndgc_compute([entry["id"] for entry in formatted_results], [entry["id"] for entry in memory["pool"]])

                        self.cum_ndcg += ndcg
                        # Update cumulative stats
                        self.total_searches += 1
                        logging.info(
                            f"Search seq {seq} completed | "
                            f"Retrieved: {len(set(entry["id"] for entry in formatted_results))}, Truth: {len(set(entry["id"] for entry in memory["pool"]))} | "
                            f"Overlap: {overlap_ration:.2f}"
                            f"Precision @5: {precision_top_5}"
                            f"NDCG: {ndcg}"
                        )
                    elif memory["op"] == "touch":
                        logging.debug(f"Executing TOUCH for seq {seq}")
                        insrt = list_for_touch(memory)
                        start = time.perf_counter_ns()
                        execute_values(self.cursor, append_database_touch_operation, [insrt])
                        self.log_times.append(time.perf_counter_ns() - start)
                        # using the k most important ones, modify in database their timestamp (id index)
                        # record metrics for the writing to database (compute metrics)

                        # modify the entries in the ids to the timestamp of "at"
                        query_modify_time = """
                            UPDATE traces
                            SET last_accessed_at = %s
                            WHERE id_operation IN %s
                        """
                        # memory["ids"] = "ids": ["a1_t2_2", "a1_t1_1", "a1_t0_2", "a1_t1_3", "a1_seed_2"]
                        ids_tuple = tuple(memory["ids"])
                        start = time.perf_counter_ns()
                        self.cursor.execute(query_modify_time, (memory["sim_t"], ids_tuple))
                        self.modification_times.append(time.perf_counter_ns() - start)

                    else:
                        logging.warning(f"Skipping unknown operation: {memory['op']}")
                        raise Exception("memory not permitted")

                # Log final metrics for this agent
                if self.total_searches > 0:
                    avg_overlap = self.cum_overlap / self.total_searches
                    avg_precision_5k = self.cum_p_at_5 / self.total_searches
                    avg_ndcg = self.cum_ndcg / self.total_searches
                    logging.info(f"--- AGENT {self.id_agent} FINAL METRICS ---")
                    logging.info(f"Avg Overlap: {avg_overlap:.4f}")
                    logging.info(f"Avg Precision 5K: {avg_precision_5k:.4f}")
                    logging.info(f"Avg Normalized Discounted Cummulative Gain: {avg_ndcg:.4f}")

                # print the p50, p95 and p99 of the insertion, search and modification times
                logging.info(f"--- AGENT {self.id_agent} TIMING METRICS ---")
                for label, samples in (
                    ("Insertion", self.insertion_times),
                    ("Search", self.search_times),
                    ("Modification", self.modification_times),
                    ("Log", self.log_times),
                ):
                    if samples:
                        p50 = float(np.percentile(samples, 50))
                        p95 = float(np.percentile(samples, 95))
                        p99 = float(np.percentile(samples, 99))
                        logging.info(
                            f"{label} Times p50 (ns): {p50:.2f}, p95 (ns): {p95:.2f}, p99 (ns): {p99:.2f}")

                logging.info("Successfully finished all operations.")
        except Exception:
            logging.exception("Agent crashed at seq %s", seq)
            raise

        finally:
            self.cursor.close()
            self.connection.close()
            logging.debug("Database connection closed.")