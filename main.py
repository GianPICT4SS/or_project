import json
import logging
import numpy as np
from simulator.instance import Instance
from solver.antenna_activation import SimpleAntennaActivation
from heuristic.simpleHeu import SimpleHeu
import networkx as net

np.random.seed(0)

class ConflictGraph():

    def __init__(self, dict_data, p):

        self.nodes = np.arange(dict_data['n_items'])
        self.p = p
        self.dict_data = dict_data
        self.Graph = net.Graph()

    def simple_conflict_graph(self, conflict=10):
        """build a simple conflict graph: without particular assumptions, a conflict graph is built,
        with a fixed number of conflict and using as antennas two"""

        while len(list(self.Graph.edges)) < conflict:

            i = np.random.randint(low=np.min(self.nodes), high=np.max(self.nodes))
            j = np.random.randint(low=np.min(self.nodes), high=np.max(self.nodes))
            #logging.info(f'(i,j) = {i,j}')
            if i != j and j+i < 2*len(self.nodes):
            # make sure that the two random nodes are different in order to create a conflict edge, without
            # taking a nodes not present in the possible nodes list.
                edge = (i, j)
                #logging.info(f'Adding edge: {edge}')
                self.Graph.add_edge(*edge)
                logging.info(f' edges: {self.Graph.edges}')

        # Take the conflict edges
        a, b = zip(*self.Graph.edges)
        self.dict_data['A'] = [x for x in a]
        self.dict_data['B'] = [x for x in b]
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

    graph = ConflictGraph(dict_data, p=0.8)
    graph.simple_conflict_graph(conflict=graph.dict_data['n_items'])

    prb = SimpleAntennaActivation()
    of_exact, sol_exact, comp_time_exact, x, prob = prb.solve(
        graph.dict_data,
        verbose=True
    )
    print(of_exact, sol_exact, comp_time_exact)

    heu = SimpleHeu(2)
    of_heu, sol_heu, comp_time_heu = heu.solve(
        dict_data
    )
    print(of_heu, sol_heu, comp_time_heu)

    # printing results of a file
    file_output = open(
        "./results/exp_general_table.csv",
        "w"
    )
    file_output.write("method, of, sol, time\n")
    file_output.write("{}, {}, {}, {}\n".format(
        "heu", of_heu, sol_heu, comp_time_heu
    ))
    file_output.write("{}, {}, {}, {}\n".format(
        "exact", of_exact, sol_exact, comp_time_exact
    ))
    file_output.close()








