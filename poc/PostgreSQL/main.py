import psycopg2
import pandas as pd
from queries import *
from agent_logic import Agent

class TypeOfDataNotFound(Exception):
    pass

TRACE_PATH = "../trace.jsonl"
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
    with psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password") as conn:
        with conn.cursor() as cursor:
            cursor.execute("""CREATE EXTENSION IF NOT EXISTS vector;""")
            cursor.execute(database_creation)  # Ensure database_creation is imported

    # Read jsonl into a DataFrame
    trace = pd.read_json(TRACE_PATH, lines=True)

    threads = []
    for agent_num in range(1, NUMBER_OF_AGENTS + 1):
        agent_string_id = f"a{agent_num}"

        agent_data_df = trace[trace['agent'] == agent_string_id]
        agent_data_list = agent_data_df.to_dict('records')

        t = Agent(agent_num, agent_data_list)
        threads.append(t)

    for t in threads:
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    initial_setup_agents_seed()