import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import psycopg2

from queries import *
from agent_logic import Agent

current_folder = Path(__file__).resolve().parent
parent_folder = current_folder.parent

# benchmark_report lives in poc/, one level up, so the Qdrant runner and this one
# summarise their numbers with the exact same code
sys.path.insert(0, str(parent_folder))
from  benchmark_report import log_summary, summarise, write_report  # noqa: E402

TRACE_PATH = parent_folder / "data/trace.jsonl"
REPORT_PATH = parent_folder / "results/postgresql.json"

NUMBER_OF_AGENTS = 100
MAX_THREADS = 16

import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO, # Sets the lowest level of logs to show (DEBUG, INFO, WARNING, ERROR)
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("database_ingestion.log"), # Saves logs to this file
        logging.StreamHandler()                        # Prints logs to the terminal
    ]
)

def initial_setup_agents_seed():
    conn = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password")
    with conn:
        with conn.cursor() as cursor:
            cursor.execute("""CREATE EXTENSION IF NOT EXISTS vector;""")
            cursor.execute(database_creation)  # Ensure database_creation is imported
    conn.close()


    # Read JSONL into a DataFrame
    agent_memories = defaultdict(list)
    with open(TRACE_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            agent = item["agent"]
            agent_memories[agent].append(item)

    agents = []
    for agent_num in range(1, NUMBER_OF_AGENTS + 1):
        agent_string_id = f"a{agent_num:03d}"

        agent_data_list = agent_memories.pop(agent_string_id, [])
        print(f"agent: {agent_string_id}, with mems: {len(agent_data_list)}")
        t = Agent(agent_num, agent_data_list)
        agents.append(t)

    if agent_memories:
        logging.warning(f"Trace holds agents with no runner: {sorted(agent_memories)}")

    failed = []
    # the clock covers the whole concurrent phase, which is what the ops/s in the
    # summary is derived from
    run_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {executor.submit(agent.run): agent for agent in agents}
        for future in as_completed(futures):
            agent = futures[future]
            try:
                future.result()
            except Exception:
                logging.exception(f"Agent {agent.id_agent} failed")
                failed.append(agent.id_agent)
    wall_clock_s = time.perf_counter() - run_start

    # a crashed agent keeps whatever samples it had taken before dying, so it
    # still contributes to the pooled percentiles; the summary names it instead
    report = summarise("PostgreSQL", agents, wall_clock_s, MAX_THREADS, failed)
    log_summary(report)
    write_report(report, REPORT_PATH)


if __name__ == "__main__":
    initial_setup_agents_seed()