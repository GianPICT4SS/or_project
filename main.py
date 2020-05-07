import json
import logging
import numpy as np
from simulator.instance import Instance
from solver.simpleKnapsack import SimpleAntennaActivation
from heuristic.simpleHeu import SimpleHeu
import networkx as net

np.random.seed(0)



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

    G = net.Graph(name='SimpleAntenna')

    ls = np.arange(dict_data['n_items'])
    for i in ls:
        edge = (i, i + 1)
        G.add_edge(*edge)

    a, b = zip(*G.edges)
    dict_data['A'] = [x for x in a]
    dict_data['B'] = [x for x in b]
    dict_data['B'][dict_data['n_items']-1] = 1



    prb = SimpleAntennaActivation()
    of_exact, sol_exact, comp_time_exact, x, prob = prb.solve(
        dict_data,
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








