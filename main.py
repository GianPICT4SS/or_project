import time
import json
import logging
import numpy as np
from simulator.instance import Instance
from solver.antenna_activation import AntennaActivation
from heuristic.simpleHeu import SimpleHeu

import matplotlib.pyplot as plt
import networkx as net

np.random.seed(0)

class ConflictGraph():

    def __init__(self, dict_data):

        self.nodes = np.arange(dict_data['n_items'])
        self.dict_data = dict_data
        self.Ograph = net.Graph()
        self.randomGraph = net.Graph()
        self.heu_graph = self.Ograph
        self.info = {}

    def simple_conflict_graph(self, conflict=10):
        """build a simple conflict graph: without particular assumptions, a conflict graph is built,
        with a fixed number of conflicts"""

        while len(list(self.Ograph.edges)) < conflict:

            i = np.random.randint(low=np.min(self.nodes), high=np.max(self.nodes)+1)
            j = np.random.randint(low=np.min(self.nodes), high=np.max(self.nodes)+1)
            #logging.info(f'(i,j) = {i,j}')
            if i != j and j+i < 2*len(self.nodes):
            # make sure that the two random nodes are different in order to create a conflict edge, without
            # taking a nodes not present in the possible nodes list.
                edge = (i, j)
                logging.info(f'Adding edge: {edge}')
                self.Ograph.add_edge(*edge)
        logging.info(f' edges: {self.Ograph.edges}')

        # Take the conflict edges
        a, b = zip(*self.Ograph.edges)
        self.dict_data['A'] = [x for x in a]
        self.dict_data['B'] = [x for x in b]
        return self

    def random_conflict_graph(self):

        N = self.dict_data['n_items']
        p = np.log(N)/N
        logging.info(f' p: {p}')

        for i in range(N):
            for j in range(N-1):
                #if i != j:
                a = np.random.randint(low=0, high=N) / N
                #logging.info(f' random a: {a}')
                if a <= p:
                    logging.info(f' random a: {a}')
                    edge = (i, j+1)
                    self.randomGraph.add_edge(*edge)
                    #logging.info(f' Random edges: {self.Graph.edges}')

        logging.info(f'Random edges: {self.randomGraph.edges}')
        # Take the conflict edges
        a, b = zip(*self.randomGraph.edges)
        self.dict_data['R_A'] = [x for x in a]
        self.dict_data['R_B'] = [x for x in b]
        return self

    def plot_graph(self, flag=False):

        plt.subplot(131)
        plt.title('Simple Conflict Graphs')
        net.draw(self.Ograph, with_labels=True, font_weight='bold')
        if flag:
            plt.subplot(132)
            plt.title('Random Conflict Graphs')
            net.draw(self.randomGraph, with_labels=True, font_weight='bold')

            plt.subplot(133)
            plt.title("Heuristic Solution")
            net.draw(self.heu_graph, with_labels=True, font_weight='bold')


    def info_graph(self):

        # Optimal
        deg_o = dict(self.Ograph.degree())
        top5_o = sorted(deg_o.items(), key=lambda x: x[1], reverse=True)[:5]

        # Random
        deg_r = dict(self.randomGraph.degree())
        top5_R = sorted(deg_r.items(), key=lambda x: x[1], reverse=True)[:5]

        deg_h = dict(self.heu_graph.degree())
        top5_h = sorted(deg_h.items(), key=lambda x: x[1], reverse=True)[:5]

        self.info['top5Degree_O'] = top5_o
        self.info['top5Degree_R'] = top5_R
        self.info['top4Degree_H'] = top5_h
        self.info['Utilities'] = self.dict_data['profits']

        return self







if __name__ == '__main__':
    log_name = "./logs/main.log"
    logging.basicConfig(
        filename=log_name,
        format='%(asctime)s %(levelname)s: %(message)s',
        level=logging.INFO, datefmt="%H:%M:%S",
        filemode='w'
    )

    with open("./etc/config.json", 'r') as f:
        sim_setting = json.loads(f.read())


    inst = Instance(
        sim_setting
    )
    dict_data = inst.get_data()

    graph = ConflictGraph(dict_data)  # Graph Initialization
    graph.simple_conflict_graph(conflict=int(graph.dict_data['n_items']))  # Simple CG
    graph.random_conflict_graph()  # Random CG

    prb = AntennaActivation()  # Solver Initialization

    # Deterministic CG Problem
    of_exact, sol_exact, comp_time_exact, x, prob = prb.solve(
        graph.dict_data,
        verbose=True
    )

    # Random CG problem
    of_exactR, sol_exactR, comp_time_exactR, xR, probR = prb.solve(
        graph.dict_data,
        verbose=True, prob_name='randomAntennaActivation', type=True
    )

    print(f"of_exact: {of_exact}\n sol_exact: {sol_exact}\n comp_time_exact: {comp_time_exact}")
    print(f"of_exactR: {of_exactR}\n sol_exactR: {sol_exactR}\n comp_time_exactR: {comp_time_exactR}")

    heu = SimpleHeu(graph=graph, n=0)
    #of_heu, sol_heu, comp_time_heu = heu.solve(
    #   dict_data
    #)
    start = time.time()
    sheu = heu.recursive_cg_solve()
    end = time.time()
    ob_heu = sheu.get_oFunction()
    print(f'Heuristic time: {end-start} \n', f'sol_heu: {ob_heu}')
    #print(of_heu, sol_heu, comp_time_heu)
    graph.info_graph()
    # printing results of a file
    with open("./results/exp_general_table.csv", "w") as f:
        f.write("method, of, sol, time\n")
        f.write(f"exact, {of_exact}, {sol_exact}, {comp_time_exact}\n")
        f.write(f"exact Random, {of_exactR}, {sol_exactR}, {comp_time_exactR}\n")
        #f.write(f"heu, {of_heu}, {sol_heu}, {comp_time_heu}")









