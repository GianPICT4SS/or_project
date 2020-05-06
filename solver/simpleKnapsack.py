# -*- coding: utf-8 -*-
import time
import logging
from pulp import *
import numpy as np


class SimpleAntennaActivation():
    def __init__(self):
        pass

    def solve(
        self, dict_data, time_limit=None,
        gap=None, verbose=False
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
        logging.info("######### SimpleKnapsack ###########")
        items = range(dict_data['n_items'])
        items = {0: (0, 1), 1: (0, 1), 2: (0, 1), 3: (0, 1), 4: (0, 1), 5: (0,1), 6: (0,1), 7: (0,1),
                 8: (0,1), 9: (0, 1), 10: (0,1), 11: (0,1), 12: (0,1), 13: (0, 1), 14: (0,1)}
        x = LpVariable.dicts(
            "X", items,
            lowBound=0,
            cat=LpInteger
        )
        # LpContinuous
        print(x.items())
        problem_name = "Antenna_Activation"

        prob = LpProblem(problem_name, LpMaximize)
        prob += lpSum([dict_data['profits'][i] * x[i] for i in items]), "obj_func"
        #prob += lpSum([dict_data['sizes'][i] * x[i] for i in items]) <= dict_data['max_size'], "max_vol"
        prob += lpSum(x[i] + x[j] for i, j in zip(*(dict_data['A'], dict_data['B']))) <= 1

        prob.writeLP("./logs/{}.lp".format(problem_name))

        msg_val = 1 if verbose else 0
        start = time.time()
        #solver = solvers.COIN_CMD(
        #    msg=msg_val,
        #    maxSeconds=time_limit,
        #    fracGap=gap
        #)
        #solver.solve(prob)
        prob.solve(solver=COIN_CMD())
        end = time.time()
        #logging.info("\t Status: {}".format(pulp.LpStatus[prob.status]))
        logging.info("\t Status: {}".format(LpStatus[prob.status]))

        sol = prob.variables()
        of = value(prob.objective)
        comp_time = end - start

        sol_x = [0] * dict_data['n_items']
        for var in sol:
            logging.info("{} {}".format(var.name, var.varValue))
            if "X_" in var.name:
                sol_x[int(var.name.replace("X_", ""))] = abs(var.varValue)
        logging.info("\n\tof: {}\n\tsol:\n{} \n\ttime:{}".format(
            of, sol_x, comp_time)
        )
        logging.info("#########")
        return of, sol_x, comp_time, x
