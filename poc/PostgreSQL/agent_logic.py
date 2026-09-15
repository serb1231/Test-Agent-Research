from multiprocessing import connection

import psycopg2
import threading

from psycopg2.extras import execute_values

from poc.PostgreSQL.main import list_for_insertion, list_for_search, list_for_touch
from queries import *
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

from psycopg import cursor


class Agent:
    # data generated is only the one that belongs to this agent
    def __init__(self, id_agent: int, data_generated):
        self.id_agent = id_agent
        self.entries_list = data_generated
        self.connection = psycopg2.connect(host="localhost", dbname="postgres", user="postgres", password="password")
        self.cursor = self.connection.cursor()

    def do_opoerations(self):
        vectors = np.load("../vectors.npy");
        try:
            with self.connection:
                with self.cursor:
                    for operation in self.entries_list:
                        if operation["op"] == "insert":
                            # add them directly to the database
                            if operation["kind"] == "observation":
                                insrt = list_for_insertion(operation)
                                execute_values(cursor, query_insert, insrt)
                            elif operation["kind"] == "reflection":
                                insrt = list_for_insertion(operation)
                                execute_values(cursor, query_insert, insrt)

                        elif operation["op"] == "search":
                            if operation["kind"] == "react":
                                insrt = list_for_search(operation)
                                execute_values(cursor, query_insert, insrt)
                                # if information important, do cosine similarity and get most important ones (vector similarity)
                                trigger = operation["trigger"]

                                # compute metric how much time it took (compute metrics)
                            elif operation["kind"] == "reflect":
                                pass
                                #  based on question, fetch 30 relevant answers (vector embedding);
                                # compute metrics for how much time it took to get the vector embeddings

                        elif operation["op"] == "touch":
                            if operation["kind"] == "react":
                                pass
                                # using the k most important ones, modify in database their timestamp (id index)
                                # recvord metrics for the writting to database (compute metrics)
                            elif operation["kind"] == "reflect":
                                pass
                                # modify the timestamp of the composite top
                                # record metrics for the modifications (compute metrics)

                        else:
                            raise Exception("operation not permitted")
        finally:
            self.connection.close()