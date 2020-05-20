# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import logging
from pulp import *

log_name = "./logs/main.log"
logging.basicConfig(
    filename=log_name,
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO, datefmt="%H:%M:%S",
    filemode='w'
)

class SimpleHeu():
    def __init__(self, n, graph):
        """

        :param n: iteration number
        :param graph: ConflictGraph object
        """
        self.n = n
        self.graph = graph
        self.tot_u = np.linalg.norm(graph.dict_data['profits'])
        self.init_CGnodes_ls = list(graph.Ograph.nodes)
        logging.info(f'init_CGnodes: {self.init_CGnodes_ls}')

    def solve(
        self, dict_data
    ):
        """[summary]
        
        Arguments:
            dict_data {[type]} -- [description]
        
        Keyword Arguments:
            time_limit {[type]} -- [description] (default: {None})
            gap {[type]} -- [description] (default: {None})
            verbose {bool} -- [description] (default: {False})
        
        Returns:
            [type] -- [description]
        """
        sol_x = [0] * dict_data['n_items']
        start = time.time()
        sol_x[self.n] = math.ceil(dict_data['max_size'] / dict_data['sizes'][0])
        end = time.time()
        
        of = dict_data['profits'][self.n] * sol_x[self.n]
        comp_time = end - start
        
        return of, sol_x, comp_time

    def recursive_cg_solve(self):
        """
        it seems to give absolutely a very good solution.
        TO DO: it is needed a method to compute the final solution obtaining as a concatenation of the final graph returned
        by this method and the nodes that are not in the CG.
        :return:
        """


        logging.info(f'heu_graph nodes: {self.graph.heu_graph.nodes}')
        #nodes_ls = list(self.graph.Graph.nodes)  # Get all nodes in CG
        u = (self.graph.dict_data['profits'][self.init_CGnodes_ls[self.n]])  # compute the Utility of the first node in the CG
        logging.info(f'profits of {self.init_CGnodes_ls[self.n]}: {u}')

        nodes_conf_n = list(self.graph.heu_graph[self.init_CGnodes_ls[self.n]])  # get all nodes in conflict with the node n
        logging.info(f'nodes conflict: {nodes_conf_n}')
        conflicts = len(nodes_conf_n)  # compute with how many nodes n is in conflict
        logging.info(f'len conflicts: {conflicts}')

        ut_conf = 0  # initialize the total utility of the nodes in conflict with n
        # compute the total utility of the nodes in confict with n
        for c in nodes_conf_n:
            ut_conf = ut_conf + self.graph.dict_data['profits'][c]
        #ut_conf = ut_conf/self.tot_u

        if ut_conf > conflicts*u:
            logging.info(f'ut_conf: {ut_conf}; conflicts*u: {conflicts*u}')
            self.graph.heu_graph.remove_node(self.init_CGnodes_ls[self.n])  # remove node n from the solution
            logging.info(f'nodes {self.init_CGnodes_ls[self.n]} turn OFF')
            # Recursion
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):  # check if all nodes have been visited
                return self.recursive_cg_solve()
            else:

                return self
        else:  # do not exclude nodes n
            # Recursion
            logging.info(f'nodes {self.init_CGnodes_ls[self.n]} turn ON')
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):
                return self.recursive_cg_solve()
            else:
                return self

    def get_oFunction(self):

        obj_function = 0
        sol_h = list(self.graph.heu_graph.nodes)
        for s in sol_h:
            obj_function += self.graph.dict_data['profits'][s]
        return obj_function




