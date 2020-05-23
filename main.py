import json
import logging
import numpy as np
from simulator.instance import Instance
from solver.antenna_activation import AntennaActivation
#from heuristic.simpleHeu import SimpleHeu
from heuristic.WGHeu import WGHeu

import matplotlib.pyplot as plt
import networkx as net

np.random.seed(0)

class ConflictGraph():

    def __init__(self, dict_data):

        self.nodes = np.arange(dict_data['n_items'])
        self.dict_data = dict_data
        self.Graph = net.Graph()
        self.randomGraph = net.Graph()

    def simple_conflict_graph(self, conflict=10):
        """build a simple conflict graph: without particular assumptions, a conflict graph is built,
        with a fixed number of conflicts"""

        while len(list(self.Graph.edges)) < conflict:

            i = np.random.randint(low=np.min(self.nodes), high=np.max(self.nodes)+1)
            j = np.random.randint(low=np.min(self.nodes), high=np.max(self.nodes)+1)
            #logging.info(f'(i,j) = {i,j}')
            if i != j and j+i < 2*len(self.nodes):
            # make sure that the two random nodes are different in order to create a conflict edge, without
            # taking a nodes not present in the possible nodes list.
                edge = (i, j)
                logging.info(f'Adding edge: {edge}')
                self.Graph.add_edge(*edge)
        logging.info(f' edges: {self.Graph.edges}')

        # Take the conflict edges
        a, b = zip(*self.Graph.edges)
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

        plt.subplot(121)
        plt.title('Simple Conflict Graphs')
        net.draw(self.Graph, with_labels=True, font_weight='bold')
        if flag:
            plt.subplot(122)
            plt.title('Random Conflict Graphs')
            net.draw(self.randomGraph, with_labels=True, font_weight='bold')







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
    graph.simple_conflict_graph(conflict=int(graph.dict_data['n_items']*0.7))  # Simple CG
    graph.random_conflict_graph()  # Random CG

    prb = AntennaActivation()  # Solver Initialization

    # Deterministic CG Problem
    of_exact, sol_exact, comp_time_exact, x, prob = prb.solve(
        graph.dict_data,
        verbose=True
    )

    #WH Heuristic
    wgheu = WGHeu()
    of_heu, sol_heu, comp_time_heu = wgheu.solve(graph.Graph, dict_data["profits"]);

    # Random CG problem
    of_exactR, sol_exactR, comp_time_exactR, xR, probR = prb.solve(
        graph.dict_data,
        verbose=True, prob_name='randomAntennaActivation', type=True
    )

    print(f"of_exact: {of_exact}\n sol_exact: {sol_exact}\n comp_time_exact: {comp_time_exact}")
    print(f"of_exactR: {of_exactR}\n sol_exactR: {sol_exactR}\n comp_time_exactR: {comp_time_exactR}")

    #heu = SimpleHeu(2)
    #of_heu, sol_heu, comp_time_heu = heu.solve(
    #   dict_data
    #)
    #print(of_heu, sol_heu, comp_time_heu)

    # printing results of a file
    with open("./results/exp_general_table.csv", "w") as f:
        f.write("method; of; time ; sol;\n")
        f.write(f"exact; {of_exact}; {comp_time_exact}; {sol_exact}\n")
        f.write(f"exact Random; {of_exactR}; {comp_time_exactR}; {sol_exactR}\n")
        f.write(f"heu; {of_heu}; {comp_time_heu}; {sol_heu}")









