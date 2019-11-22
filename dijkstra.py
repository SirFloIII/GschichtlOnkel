# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 19:23:02 2018

@author: Flo
"""

def dijkstra(start, goal, neighbors, cost = lambda x,y:1 , heuristic = lambda x,y:0):
    """
    search minimal path
    nodes must be hashable
    
    start is node
    goal is node or set
    neighbors is a function of node that returns list of neighbor nodes
    cost is function of two nodes
    heuristic is function of two nodes, transforms algorithm into A*
    only use heuristic if not in set mode
    """    
    
    setmode = True if isinstance(goal, set) else False
    funcmode = True if callable(goal) else False
    
    dist = dict()
    prev = dict()
    
    Q = set()
    notQ = set()
    
    Q.add(start)
    for n in neighbors(start):
        Q.add(n)
        
    for v in Q:
        dist[v] = float("inf")
        prev[v] = None
    dist[start] = 0
    
    while len(Q):
        u = min(Q, key = lambda x: dist[x] + heuristic(x, goal))
        
        Q.remove(u)
        notQ.add(u)
        
        if (funcmode and goal(u)) or (setmode and u in goal) or (
                      not setmode and not funcmode and u == goal):
            path = []
            while not prev[u] == None:
                path.append(u)
                u = prev[u]
            path.append(u)
            path.reverse()
            return path
            
        for v in neighbors(u):
            if v not in notQ and v not in Q:
                Q.add(v)
                dist[v] = float("inf")
                prev[v] = None
            
            alt = dist[u] + cost(u, v)
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    
    print("Warning: No Path found")
    return []

def dijkstra2(start, goalset, graph, cost = lambda x,y:1):
    
    dist = dict()
    prev = dict()
    
    Q = graph.vertices.copy()
    
    assert start in Q
    
    for v in Q:
        dist[v] = float("inf")
        prev[v] = None
    dist[start] = 0
    
    while Q:
        print(len(Q))
        u = min(Q, key = lambda x: dist[x])
        
        if dist[u] == float("inf"):
            print("Warning: No Path found")
            return []
        
        Q.remove(u)
        
        if u in goalset:
            path = []
            while prev[u] != None:
                path.append(u)
                u = prev[u]
            path.append(u)
            path.reverse()
            return path
        
        for v in graph.edges[u]:
            alt = dist[u] + cost(u, v)
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
    
    print("Warning: No Path found and experiencing unexpected behavior")
    return []