# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import logging
from pulp import *

from queue import Queue

Q = Queue()

log_name = "./logs/main.log"
logging.basicConfig(
    filename=log_name,
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO, datefmt="%H:%M:%S",
    filemode='w'
)

class SimpleHeu():
    def __init__(self, n, graph, dict_data):
        """

        :param n: iteration number
        :param graph: ConflictGraph object
        """
        self.n = n
        self.graph = graph
        self.dict_data = dict_data
        self.nodes_deleted = Queue()
        self.tot_u = np.linalg.norm(dict_data['profits'])
        self.init_CGnodes_ls = list(graph.nodes)
        #self.init_CGnodes_ls = np.unique(graph.dict_data['R_A'] + graph.dict_data['R_B'])
        logging.info(f'init_CGnodes: {self.init_CGnodes_ls}')


    def recursive_cg_solve(self, mu=1):
        """
        it seems to give absolutely a very good solution.
        TO DO: it is needed a method to compute the final solution obtaining as a concatenation of the final graph returned
        by this method and the nodes that are not in the CG.
        :return:
        """


        logging.info(f'heu_graph nodes: {self.graph.nodes}')
        #nodes_ls = list(self.graph.Graph.nodes)  # Get all nodes in CG
        u = (self.dict_data['profits'][self.init_CGnodes_ls[self.n]])  # compute the Utility of the first node in the CG
        logging.info(f'profits of {self.init_CGnodes_ls[self.n]}: {u}')

        try:
            nodes_conf_n = list(self.graph[self.init_CGnodes_ls[self.n]])  # get all nodes in conflict with the node n
            logging.info(f'nodes conflict: {nodes_conf_n}')
            logging.info(f"utilities conflict nodes: {self.dict_data['profits'][nodes_conf_n]}")
            conflicts = len(nodes_conf_n)  # compute with how many nodes n is in conflict
            logging.info(f'len conflicts: {conflicts}')
        except:
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):
                return self.recursive_cg_solve()
            else:
                return self

        #nodes_conf_n_utilities = self.dict_data['profits'][nodes_conf_n]

        ut_conf = 0  # initialize the total utility of the nodes in conflict with n
        # compute the total utility of the nodes in confict with n
        for c in nodes_conf_n:
            ut_conf = ut_conf + self.dict_data['profits'][c]
        #ut_conf = ut_conf/self.tot_u

        if ut_conf > u*conflicts*mu:
            logging.info(f'ut_conf: {ut_conf}; conflicts*u*mu: {conflicts*u*mu}')
            self.graph.remove_node(self.init_CGnodes_ls[self.n])  # remove node n from the solution
            self.nodes_deleted.put(self.init_CGnodes_ls[self.n])
            logging.info(f'nodes {self.init_CGnodes_ls[self.n]} turn OFF')
            # Recursion
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):  # check if all nodes have been visited
                return self.recursive_cg_solve()
            else:
                self.dict_data['Heu_sol'] = np.sort(list(self.graph.nodes))
                return self
        else:  # do not exclude nodes n but others

            self.graph.remove_nodes_from(nodes_conf_n)
            self.nodes_deleted.put(nodes_conf_n)
            # Recursion
            logging.info(f'nodes {nodes_conf_n} turn OFF')
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):
               return self.recursive_cg_solve()
            else:
                self.dict_data['Heu_sol'] = np.sort(list(self.graph.nodes))
                return self

    def get_oFunction(self):

        obj_function = 0
        sol_h = list(self.graph.nodes)
        for s in sol_h:
            obj_function += self.dict_data['profits'][s]
        return obj_function

    def join_variables(self):

        items = list(range(self.dict_data['n_items']))
        Q.put(items)

        deleted = Q.get(self.nodes_deleted)

        final_sol = Q.queue

        return final_sol




