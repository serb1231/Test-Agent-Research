import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qdrant_client import QdrantClient, models

from collections_config import (
    MEMORIES_COLLECTION,
    OPS_COLLECTION,
    SEARCH_PARAMS,
    agent_filter,
    ids_filter,
    memory_payload,
    search_payload,
    touch_payload,
)

current_folder = Path(__file__).resolve().parent
parent_folder = current_folder.parent

VECTORS_PATH = parent_folder / "data/vectors.npy"

QDRANT_URL = "http://localhost:6333"

# Without this an idle HTTP request can park a pool worker forever, and since the
# pool only has MAX_THREADS slots a few of those would stall the whole run.
REQUEST_TIMEOUT_S = 60

# Mapped once at import instead of once per agent. The file is 600 MB and every
# agent reads from the same pages, so 100 separate maps bought nothing; a numpy
# memmap is only ever read here, which is safe to share across the pool threads.
VECTORS: npt.NDArray[np.float32] = np.load(VECTORS_PATH, mmap_mode="r")


# NDGC (Normalised Discounted Cummulative Gain)
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
        self.client = None
        self.id_agent = id_agent
        self.entries_list = data_generated
        logging.info(f"Agent {id_agent}: initialized with {len(self.entries_list)} operations to process.")
        self.insertion_times: list[int] = []
        self.log_times: list[int] = []
        self.modification_times: list[int] = []
        self.search_times: list[int] = []

        # metrics for searching database
        self.total_searches = 0
        self.cum_overlap = 0.0
        self.cum_ndcg = 0.0
        self.cum_p_at_5 = 0.0

    def run(self):
        logging.info(f"Agent {self.id_agent}: started processing operations.")
        # One client per agent, the same way the PG version opens one connection
        # per agent. Binding it before the try matters: a client that fails to
        # construct must not reach the finally, or close() on None would mask the
        # real error. Each client is reached by exactly one pool worker and is
        # closed before that worker is handed the next agent, so the httpx
        # connection pool underneath is never shared across threads.
        client = QdrantClient(url=QDRANT_URL, timeout=REQUEST_TIMEOUT_S)
        self.client = client
        seq = None
        try:
            for memory in self.entries_list:
                seq = memory.get('seq')
                if memory["op"] == "insert":
                    logging.debug(f"Executing INSERT for seq {seq}")
                    point = models.PointStruct(
                        id=seq,  # seq is unique across the whole trace
                        vector=VECTORS[int(memory["vec"])].tolist(),
                        payload=memory_payload(memory),
                    )
                    start = time.perf_counter_ns()
                    # wait=True is the analogue of PG's autocommit: the call only
                    # returns once the write is applied, otherwise we would be
                    # timing a fire-and-forget
                    self.client.upsert(
                        collection_name=MEMORIES_COLLECTION, points=[point], wait=True
                    )
                    self.insertion_times.append(time.perf_counter_ns() - start)

                elif memory["op"] == "search":
                    logging.debug(f"Executing SEARCH for seq {seq}")
                    log_point = models.PointStruct(
                        id=seq, vector={}, payload=search_payload(memory)
                    )
                    start = time.perf_counter_ns()
                    self.client.upsert(
                        collection_name=OPS_COLLECTION, points=[log_point], wait=True
                    )
                    self.log_times.append(time.perf_counter_ns() - start)

                    # get the vector of the trigger
                    embedding_of_trigger = VECTORS[int(memory["qvec"])].tolist()

                    start = time.perf_counter_ns()
                    response = self.client.query_points(
                        collection_name=MEMORIES_COLLECTION,
                        query=embedding_of_trigger,
                        query_filter=agent_filter(memory["agent"]),
                        limit=int(memory["k_prime"]),
                        with_payload=["id_operation", "memory_text"],
                        search_params=SEARCH_PARAMS,
                    )
                    self.search_times.append(time.perf_counter_ns() - start)

                    # cosine distance -> score is the similarity, already best first
                    raw_results: list[Any] = response.points
                    formatted_results = []
                    for hit in raw_results:
                        formatted_results.append({
                            "id": hit.payload["id_operation"],
                            "text": hit.payload["memory_text"],
                            "cosine": hit.score,
                        })

                    overlap_ration = overlap(formatted_results, memory)
                    self.cum_overlap += overlap_ration

                    precision_top_5 = precision_5(formatted_results, memory)
                    self.cum_p_at_5 += precision_top_5

                    ndcg = ndgc_compute(
                        [entry["id"] for entry in formatted_results],
                        [entry["id"] for entry in memory["pool"]],
                    )
                    self.cum_ndcg += ndcg

                    # Update cumulative stats
                    self.total_searches += 1
                    logging.info(
                        f"Search seq {seq} completed | "
                        f"Retrieved: {len(set(entry['id'] for entry in formatted_results))}, "
                        f"Truth: {len(set(entry['id'] for entry in memory['pool']))} | "
                        f"Overlap: {overlap_ration:.2f} "
                        f"Precision @5: {precision_top_5} "
                        f"NDCG: {ndcg}"
                    )

                elif memory["op"] == "touch":
                    logging.debug(f"Executing TOUCH for seq {seq}")
                    log_point = models.PointStruct(
                        id=seq, vector={}, payload=touch_payload(memory)
                    )
                    start = time.perf_counter_ns()
                    self.client.upsert(
                        collection_name=OPS_COLLECTION, points=[log_point], wait=True
                    )
                    self.log_times.append(time.perf_counter_ns() - start)

                    # modify the entries in the ids to the timestamp of "at"
                    # memory["ids"] = ["a001_t2_2", "a001_t1_1", ...]
                    start = time.perf_counter_ns()
                    self.client.set_payload(
                        collection_name=MEMORIES_COLLECTION,
                        payload={"last_accessed_at": memory["sim_t"]},
                        points=ids_filter(memory["ids"]),
                        wait=True,
                    )
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
            client.close()
            logging.debug("Qdrant client closed.")
