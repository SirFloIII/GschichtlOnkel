# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 00:11:25 2019

@author: Flo
"""

import numpy as np
import networkx as nx
import PIL
from matplotlib import pyplot as plt
from tqdm import tqdm


def weightedAvg(a, a_w, b, b_w):
    a_w, b_w = a_w/(a_w+b_w), b_w/(a_w+b_w)
    return a*a_w + b*b_w

def weightedColorAvg(a, a_w, b, b_w):
    return np.sqrt(weightedAvg(a*a, a_w, b*b, b_w))

class Node:
    
    def __init__(self, color, pixelPos, pixels):
        
        self.color = color.astype(np.int32)
        self.pixelPos = np.array(pixelPos)
        self.pixels = pixels
        self.n = len(pixels)
        
        self.id = tuple(pixelPos)
        
    def mergeWith(self, node, G):
        
        assert G.has_edge(self, node)
        
        self.color = weightedColorAvg(self.color, self.n, node.color, node.n)
        self.pixelPos = weightedAvg(self.pixelPos, self.n, node.pixelPos, node.n)
        self.pixels = self.pixels + node.pixels
        self.n += node.n
        
        neighbors = [x[1] for x in G.edges((node))]
        
        for n in neighbors:
            if n != self:
                G.add_edge(self, n)
        
        G.remove_node(node)
        
    @classmethod
    def fromPixel(cls, color, pixelPos):
        
        return cls(color, pixelPos, [pixelPos])
    
    def __hash__(self):
        
        return hash(self.id)
        
def graphifyImageFile(path):

    image = PIL.Image.open(path)
    
    return graphifyImage(image)
    
def graphifyImage(image):
    
    array = np.array(image.convert())

    return graphifyArray(array)

def graphifyArray(array):
    
    G = nx.Graph()
    
    indexToNode = dict()
    
    for i, row in tqdm(enumerate(array)):
        for j, pixel in enumerate(row):
            n = Node.fromPixel(pixel, (i,j))
            G.add_node(n)
            indexToNode[(i,j)] = n
            if i != 0:
                G.add_edge(n, indexToNode[(i - 1, j)])
            if j != 0:
                G.add_edge(n, indexToNode[(i, j - 1)])
        
    return G, array.shape

def drawGraph(G, drawEdges = True):
    
    plt.figure()
    plt.subplot(111)
    if drawEdges:
        
        edges = G.edges()
        
        _s = [ x[0].pixelPos[1] for x in edges]
        _t = [-x[0].pixelPos[0] for x in edges]
        _u = [ x[1].pixelPos[1] for x in edges]
        _v = [-x[1].pixelPos[0] for x in edges]
        x = []
        y = []
        for s, t, u, v in zip(_s, _t, _u, _v):
            x.append(s)
            x.append(u)
            x.append(None)
            y.append(t)
            y.append(v)
            y.append(None)
        
        plt.plot(x, y, c = "black", zorder = 1)
    
    x = [ n.pixelPos[1] for n in G.nodes()]
    y = [-n.pixelPos[0] for n in G.nodes()]
    c = [n.color/256 for n in G.nodes()]
    
    plt.scatter(x, y, c = c, alpha = 1, zorder = 2)


maxD = np.linalg.norm([256**2]*3)
def similar(n, m, tol):
    return np.linalg.norm((n.color**2-m.color**2)) <= tol*maxD

def reduceGraph(G, tol):
    
    Done = False
    while not Done:
        print(f"{len(G.nodes())} remaining.")    
        #for i in range(1):    
        Done = True
        
        edgesCopy = G.edges()
        bannedNodes = []
        
        for edge in edgesCopy:
            if G.has_edge(*edge):
                n = edge[0]
                m = edge[1]
#                assert G.has_node(n)
#                assert G.has_node(m)
                if similar(n, m, tol) and n not in bannedNodes and m not in bannedNodes:
                    n.mergeWith(m, G)
                    bannedNodes.append(n)
                    Done = False
#                    assert not G.has_node(m)
#                    assert G.has_node(n)
    print("")
    return G



#path = "kleineNarzisse.jpg"
#path = "smallTestImage.png"
#path = "schachbrett.png"
path = "tulpe.jpg"

print("graphifiying")
G, shape = graphifyImageFile(path)


print("reducing")
tol = 0.05
G = reduceGraph(G, tol)

#print(set([tuple(x.color) for x in G.nodes()]))                
print("drawing")
drawGraph(G, drawEdges=True)


print("reconstructing")
#def reconstructImage(G, shape):
img = np.ones(shape, dtype = np.uint8)

for n in G.nodes():
    for pixel in n.pixels:
        img[pixel] = n.color
        
PIL.Image.fromarray(img).resize((shape[1]*5, shape[0]*5)).save(f"savedImages/tol_{tol}_{path}")














