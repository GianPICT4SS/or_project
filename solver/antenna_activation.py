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
        logging.info("######### Simple Antenna Activation ###########")
        items = range(dict_data['n_items'])


        x = LpVariable.dicts(
            "X", items,
            lowBound=0,
            upBound=1,
            cat=LpInteger
        )

        # LpContinuous
        print(x.items())
        problem_name = "Antenna_Activation"

        prob = LpProblem(problem_name, LpMaximize)
        prob += lpSum([dict_data['profits'][i] * x[i] for i in items]), "obj_func"
        # prob += lpSum([pro[i] * x[i] for i in items]), "obj_func"
        # prob += lpSum([dict_data['sizes'][i] * x[i] for i in items]) <= dict_data['max_size'], "max_vol"
        for i in range(len(dict_data['A'])):
            """Do not sum all constraints: splits constraints"""
            # prob += lpSum(x[i] + x[j] for i, j in zip(*(dict_data['A'], dict_data['B']))) <= 1, "max_vol"
            k = dict_data['A'][i]
            j = dict_data['B'][i]
            prob += lpSum(x[k] + x[j]) <= 1, f"conflict_{i}"

        logging.info(f'profits: {dict_data["profits"]}')
        prob.writeLP("./logs/{}.lp".format(problem_name))

        msg_val = 1 if verbose else 0
        start = time.time()
        prob.solve(solver=COIN_CMD())
        end = time.time()
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
        return of, sol_x, comp_time, x, prob