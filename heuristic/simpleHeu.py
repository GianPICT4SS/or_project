# -*- coding: utf-8 -*-

import numpy as np
import logging
import time




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
        self.graph = graph.copy()
        self.dict_data = dict_data.copy()
        self.obj_func = []
        self.solution = []
        self.nodes_deleted = []
        self.tot_u = np.linalg.norm(dict_data['profits'])
        self.init_CGnodes_ls = list(graph.nodes)
        #self.init_CGnodes_ls = np.unique(graph.dict_data['R_A'] + graph.dict_data['R_B'])
        logging.info(f'init_CGnodes: {self.init_CGnodes_ls}')


    def recursive_cg_solve(self, mu=1):
        """
        An heuristic algorithm for solving:
        max. sum_i[(a_i)*x_i)]
        s.t  x_i + x_j <= 1 if (i,j) in CF, where CF is the Conflict Graph for the problem.



        :return:
        """


        logging.info(f'heu_graph nodes: {self.graph.nodes}')
        #nodes_ls = list(self.graph.Graph.nodes)  # Get all nodes in CG
        u = (self.dict_data['profits'][self.init_CGnodes_ls[self.n]])  # compute the Utility of the first node in the CG
        logging.info(f'profits of {self.init_CGnodes_ls[self.n]}: {u}')

        try:
            # if self.init_CGnodes_ls[self.n] is present in CG take its conflict nodes
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
        ut_conf = 0  # initialize the total utility of the nodes in conflict with n
        # compute the total utility of the nodes in confict with n
        for c in nodes_conf_n:
            ut_conf = ut_conf + self.dict_data['profits'][c]

        if ut_conf > u*conflicts*mu:
            # remove node n from the final solution
            logging.info(f'ut_conf: {ut_conf}; conflicts*u*mu: {conflicts*u*mu}')
            self.graph.remove_node(self.init_CGnodes_ls[self.n])  # remove node n from the solution
            self.nodes_deleted.append([self.init_CGnodes_ls[self.n]])
            logging.info(f'nodes {self.init_CGnodes_ls[self.n]} turn OFF')
            # Recursion
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):  # check if all nodes have been visited
                return self.recursive_cg_solve()
            else:
                return self
        else:  # do not exclude nodes n, so delete other

            self.graph.remove_nodes_from(nodes_conf_n)
            self.nodes_deleted.append(nodes_conf_n)
            # Recursion
            logging.info(f'nodes {nodes_conf_n} turn OFF')
            self.n = self.n + 1
            if self.n < len(self.init_CGnodes_ls):
               return self.recursive_cg_solve()
            else:
               return self

    def get_oF_sol(self):
        """
        Get obj_function and solution of the problem
        :return: obj_function, solution
        """

        items = list(range(self.dict_data['n_items']))

        deleted = [item for sublist in self.nodes_deleted for item in sublist]

        solution = [item for item in items if item not in deleted]

        obj_function = 0
        for s in solution:
            obj_function += self.dict_data['profits'][s]
        self.obj_func = obj_function
        self.solution = solution
        return self

class MWIS():

    def __init__(self, graph, dict_data):

        self.graph = graph.copy()
        self.dict_data = dict_data.copy()
        self.ob_func = None
        self.solution = list()
        self.init_cg = np.sort(list(graph.nodes)).copy()
        self.weight = list()
        self.comp_time = None


    def mwis_dp(self):
        """
        * A Dynamic Programming heuristic algorithm for solving:
        max. sum_i[(a_i)*x_i)]
        s.t  x_i + x_j <= 1 if (i,j) in CF, where CF is the Conflict Graph for the problem.

        complexity O(n) where n is the number of nodes in the CG

        :return:
        """

        logging.info('MWIS DP solver started!')
        start = time.time()
        #self.solution.clear()
        #nVertex = len(self.init_cg)
        #logging.info(f'Number of vertex in the initial CG: {nVertex}')
        #logging.info(f"init_cn: {self.init_cg}")
        it_cg = list(self.init_cg)
        for i in it_cg:
            # skip node already in the solution
            #if self.solution.count([i]) != 0:
            #    break
            #logging.info(f" tested node: {i}")
            try:

                    conf_nodes = list(self.graph[i])
                    if len(conf_nodes) == 0:
                        continue
                    #logging.info(f"conf nodes of {i}: {conf_nodes}")
                    u_tot = [self.dict_data['profits'][x] for x in conf_nodes]
                    u_tot.append(self.dict_data['profits'][i])
                    #logging.info(f"utility node {i}: {u_tot[-1]}")
                    #logging.info(f"total utility: {u_tot}")
                    max_u = np.max(u_tot)
                    #logging.info(f'Max Utility: {max_u}')
                    # get node with utility equal to max_u
                    opt_node = list(self.dict_data['profits']).index(max_u)
                    conf_nodes.append(i)
                    delete_nodes = list(filter(lambda x: x != opt_node, conf_nodes))
                    #logging.info(f"optimal node: {opt_node}, with utility: {self.dict_data['profits'][opt_node]};"
                    #         f" \n deleted_nodes: {delete_nodes}")
                    # delete not optimal nodes
                    self.graph.remove_nodes_from(delete_nodes)
                    #self.solution.append(opt_node)
                    #logging.info(f"solution: {self.solution}")
            except:

                    pass


        elapsed = time.time() - start
        self.comp_time = elapsed
        #logging.info(f"=== Optimization done === \n Elapsed time: {elapsed} [ms] \n nodes: {len(self.solution)}")

        self.ob_func = np.sum([self.dict_data['profits'][x] for x in list(self.graph.nodes)])

        return self




















