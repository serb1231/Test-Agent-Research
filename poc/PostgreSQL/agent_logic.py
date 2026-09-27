import logging
import time
from typing import Any

import psycopg2
import threading
import json
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from queries import *
import numpy as np
import numpy.typing as npt

from psycopg import cursor


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
        item.get("at")
    )


class Agent(threading.Thread):
    # data generated is only the one that belongs to this agent
    def __init__(self, id_agent: int, data_generated):
        super().__init__()
        self.id_agent = id_agent
        self.entries_list = data_generated
        self.connection = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password")
        self.cursor = self.connection.cursor()
        logging.info(f"Initialized with {len(self.entries_list)} operations to process.")
        self.cum_insertion_time = 0
        self.cum_modif_time = 0
        self.cum_search_time = 0

        # metrics for searching database
        self.total_searches = 0
        self.cum_precision = 0.0
        self.cum_recall = 0.0

    def run(self):
        vectors : npt.NDArray[np.float32] = np.load("../vectors.npy")
        logging.info("Started processing operations.")
        try:
            register_vector(self.connection)
            with self.connection:
                with self.cursor:
                    for memory in self.entries_list:
                        seq = memory.get('seq')
                        if memory["op"] == "insert":
                            logging.debug(f"Executing INSERT for seq {seq}")
                            # add them directly to the database
                            insrt = list_for_insertion(memory, vectors[int(memory["vec"])].tolist())
                            start = time.time_ns()
                            execute_values(self.cursor, append_database_insert_operation, [insrt])
                            self.cum_insertion_time += time.time_ns() - start

                        elif memory["op"] == "search":
                            logging.debug(f"Executing SEARCH for seq {seq}")
                            insrt = list_for_search(memory)
                            start = time.time_ns()
                            execute_values(self.cursor, append_database_search_operation, [insrt])
                            self.cum_insertion_time += time.time_ns() - start
                            # get the vector of the trigger
                            embedding_of_trigger = vectors[int(memory["qvec"])].tolist()

                            query_fetch_based_on_cosine = """
                                                SELECT id_operation, memory_text, 1 - (embedding <=> %s::vector) AS similarity
                                                FROM traces
                                                WHERE embedding is NOT NULL AND op = 'insert' AND agent = '%s'
                                                ORDER BY embedding <=> %s::vector
                                                LIMIT %s
                            """
                            start = time.time_ns()
                            self.cursor.execute(query_fetch_based_on_cosine, (embedding_of_trigger,self.id_agent, embedding_of_trigger, int(memory["k_prime"])))
                            self.cum_search_time += time.time_ns() - start

                            raw_results: list[tuple[Any, ...]] = self.cursor.fetchall()

                            formatted_results = []
                            for row in raw_results:
                                formatted_results.append({
                                    "id" : row[0],
                                    "text": row[1],
                                    "cosine": row[2]
                                    })

                            sequences_retrieved = set(entry["id"] for entry in formatted_results)
                            sequences_ground_truth = set(entry["id"] for entry in memory["pool"])

                            # Calculate True Positives (overlap)
                            true_positives = len(sequences_retrieved.intersection(sequences_ground_truth))

                            # Precision: (Correct Retrieved) / (Total Retrieved)
                            precision = true_positives / len(sequences_retrieved) if sequences_retrieved else 0.0

                            # Recall: (Correct Retrieved) / (Total Ground Truth)
                            recall = true_positives / len(sequences_ground_truth) if sequences_ground_truth else 0.0

                            # Update cumulative stats
                            self.total_searches += 1
                            self.cum_precision += precision
                            self.cum_recall += recall

                            logging.info(
                                f"Search seq {seq} completed | "
                                f"Retrieved: {len(sequences_retrieved)}, Truth: {len(sequences_ground_truth)} | "
                                f"Precision: {precision:.2f} - Recall: {recall:.2f}"
                            )
                        elif memory["op"] == "touch":
                            logging.debug(f"Executing TOUCH for seq {seq}")
                            insrt = list_for_touch(memory)
                            start = time.time_ns()
                            execute_values(self.cursor, append_database_touch_operation, [insrt])
                            self.cum_insertion_time += time.time_ns() - start
                            # using the k most important ones, modify in database their timestamp (id index)
                            # record metrics for the writing to database (compute metrics)

                            # modify the entries in the ids to the timestamp of "at"
                            query_modify_time = """
                                UPDATE traces
                                SET created_at = %s
                                WHERE id_operation IN %s
                            """
                            # memory["ids"] = "ids": ["a1_t2_2", "a1_t1_1", "a1_t0_2", "a1_t1_3", "a1_seed_2"]
                            ids_tuple = tuple(memory["ids"])
                            start = time.time_ns()
                            self.cursor.execute(query_modify_time, (memory["at"], ids_tuple))
                            self.cum_modif_time += time.time_ns() - start

                        else:
                            logging.warning(f"Skipping unknown operation: {memory['op']}")
                            raise Exception("memory not permitted")

                    # Log final metrics for this agent
                    if self.total_searches > 0:
                        avg_precision = self.cum_precision / self.total_searches
                        avg_recall = self.cum_recall / self.total_searches
                        logging.info(f"--- AGENT {self.id_agent} FINAL METRICS ---")
                        logging.info(f"Avg Precision: {avg_precision:.4f} | Avg Recall: {avg_recall:.4f}")

                    # Convert nanoseconds to milliseconds for easier reading
                    logging.info(f"Total Insert Time: {self.cum_insertion_time / 1_000_000:.2f} ms")
                    logging.info(f"Total Search Time: {self.cum_search_time / 1_000_000:.2f} ms")
                    logging.info(f"Total Touch Time:  {self.cum_modif_time / 1_000_000:.2f} ms")
                    logging.info("Successfully finished all operations.")

                    logging.info("Successfully finished all operations.")
        except Exception as e:
            # exc_info=True automatically prints the full traceback in the log!
            logging.error(f"Agent crashed while processing seq {memory.get('seq')}", exc_info=True)

        finally:
            self.cursor.close()
            self.connection.close()
            logging.debug("Database connection closed.")