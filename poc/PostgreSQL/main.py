import json
from pathlib import Path

import psycopg2
from queries import *
from agent_logic import Agent

class TypeOfDataNotFound(Exception):
    pass

current_folder = Path(__file__).resolve().parent
parent_folder = current_folder.parent

TRACE_PATH = parent_folder / "trace.jsonl"

NUMBER_OF_AGENTS = 2

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
    with open(TRACE_PATH, "r") as f:
        trace = [json.loads(l) for l in f]

    threads = []
    for agent_num in range(1, NUMBER_OF_AGENTS + 1):
        agent_string_id = f"a{agent_num}"

        agent_data_list = [item for item in trace if item.get('agent') == agent_string_id]

        t = Agent(agent_num, agent_data_list)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    initial_setup_agents_seed()