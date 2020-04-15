# -*- coding: utf-8 -*-
import time
import math
import logging
from pulp import *

class SimpleHeu():
    def __init__(self, n):
        self.n = n

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
