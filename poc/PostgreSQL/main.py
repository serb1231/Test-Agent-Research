import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg2

from poc.PostgreSQL.agent_logic import Agent
from queries import *
from agent_logic import Agent

class TypeOfDataNotFound(Exception):
    pass

current_folder = Path(__file__).resolve().parent
parent_folder = current_folder.parent

TRACE_PATH = parent_folder / "data/trace.jsonl"

NUMBER_OF_AGENTS = 100
MAX_THREADS = 16

def agent_do_work(id_agent: int):
    print(f"hello from agent {id_agent}")


import logging

# Configure the logger
logging.basicConfig(
    level=logging.INFO, # Sets the lowest level of logs to show (DEBUG, INFO, WARNING, ERROR)
    format='%(asctime)s - [Agent %(threadName)s] - %(levelname)s - %(message)s',
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


    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for agent in agents:
            executor.submit(agent.run)


if __name__ == "__main__":
    initial_setup_agents_seed()