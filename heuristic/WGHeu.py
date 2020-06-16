# -*- coding: utf-8 -*-
import time
import networkx as nx
import numpy as np
#paper --> http://www.ru.is/~mmh/papers/WIS_WG.pdf

class WGHeuNode():

    def __init__(self, node_index, weighted_degree):
        self.node_index = node_index
        self.weighted_degree = weighted_degree

    def __lt__(self, other):
         return self.weighted_degree < other.weighted_degree        
        

class WGHeu():

    def solve(
        self, graph, profits
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
        start = time.time()
        #compute the adjacency matrix
        adj = nx.adj_matrix(graph).todense()

        #-1 not considered, 1 choose, 0 discard
        IS = -np.ones(adj.shape[0])

        #weights
        weights = np.array([profits[u] for u in graph])
        weighted_degrees = weights.dot(adj!=0)/weights

        #support data structure
        wgnodes = []
        for i in range(len(graph.nodes)):
            wgnodes.append(WGHeuNode(i, weighted_degrees[0,i]))
        #sort the nodes by the weighted degree
        wgnodes.sort()

        wgnode_index = 0

        while np.any(IS==-1):
            selected_node = wgnodes[wgnode_index]
            selected_node_index = selected_node.node_index      
            if IS[selected_node_index] != -1:
                wgnode_index = wgnode_index + 1
                continue                          
            IS[selected_node_index] = 1 #choose the node
            neighbors = np.argwhere(adj[selected_node_index,:]!=0) #find neighbors and excludes them from the solution
            if neighbors.shape[0]:
                IS[neighbors] = 0
            wgnode_index = wgnode_index + 1 #go to the next node in the sorted list 
        end = time.time()
        of = weights.dot(IS) #compute the value of the objective function
        return of, IS.tolist(), end - start,        


