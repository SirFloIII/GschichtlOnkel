import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image


class Region:

    def __init__(self, point, array):
        self.array_shape = array.shape
        self.min_idx = point
        self.max_idx = None
        self.active = True
        self.points = set()     #points, both inner and on the edge
        self.edge = set()       #border, belonging to the region
        self.halo = set()       #border, not (yet) part of the region
        self.saddles = set()    #saddles needn't be a point in the region
        self.add(point)

    def __repr__(self):
        return str(self.points)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        yield self.points

    def plot(self):
        assert len(self.array_shape)==2
        
        points = np.array(list(self.points)).T
        plt.scatter(points[0], points[1],
                    c="k", marker=".", s=5)

        edge = np.array(list(self.edge)).T
        plt.scatter(edge[0], edge[1],
                    c="g", marker="o", s=10)

        halo = np.array(list(self.halo)).T
        plt.scatter(halo[0], halo[1],
                    c="0.5", marker="o", s=10)
        
        plt.scatter(self.min_idx[0], self.min_idx[1],
                    c="b", marker="v", s=30)

        if self.max_idx:
            plt.scatter(self.max_idx[0], self.max_idx[1],
                        c="r", marker="^", s=30)
        
        if self.sad_idx:
            plt.scatter(self.sad_idx[0], self.sad_idx[1],
                        c="k", marker="x", s=30)
        
        plt.show()

    #get points with index varying at most by 1 in every dimension
    def get_cube(self, point):
        idx = np.array(point)
        low_corner = idx-1
        low_corner[low_corner<0] = 0
        high_corner = np.minimum(idx+1, np.array(self.array_shape)-1)
        offsets = [np.array(i)
                   for i in np.ndindex(tuple(high_corner+1-low_corner))]
        return {tuple(low_corner+o) for o in offsets}

    #get points that are directly adjacent along the axes
    def get_neigh(self, point):
        neigh = set()
        for dim, len in enumerate(self.array_shape):
            pos = point[dim]
            for k in [max(0, pos-1), pos, min(len-1, pos+1)]:
                new_idx = tuple(p if i!=dim else k for i,p in enumerate(point)) 
                neigh.add(new_idx)
        return neigh

    def add(self, point):
        assert point not in self.points
        assert point in self.halo or not self.points

        neigh = self.get_neigh(point)
        self.points.add(point)

        new_halo = neigh.difference(self.points)
##        new_halo = {p for p in new_halo
##                    if self.array[p]>= self.array[point]}

        self.halo.update(new_halo)
        self.halo.discard(point)
        
        self.edge.add(point)
        for ned in neigh.intersection(self.edge):
            if self.get_neigh(ned).issubset(self.points):
                self.edge.remove(ned)
                
        if not new_halo:
            #we found a maximum
            self.max_idx = point
            self.edge.difference_update(neigh)
            self.passivate()

    def get_edge(self):
        return self.edge

    def get_halo(self):
        return self.halo

    def remove_from_halo(self, points):
        #TODO or scrap if not needed
        pass

    def set_saddle(self, point):
        self.saddles.add(point)

    def passivate(self):
        self.active = False


class SlopeDecomposition:

    def __init__(self, array):
        self.array = array
        self.active_regions = []
        self.passive_regions = []
        self.regions = []
        
        #sort indices for increasing array value
        sorted_idx = np.unravel_index(self.array.argsort(axis=None),
                                      self.array.shape)

        #create empty levelsets for each value that occurs
        levelset = {val : set() for val in self.array[sorted_idx]}

        #then fill in the indices
        for idx in np.array(sorted_idx).T:
            levelset[self.array[tuple(idx)]].add(tuple(idx))

        for level, points in levelset.items():
            #remember which points we add to which region
            added_points = {}

            #first off, deal with points that can be assigned to existing regions
            for region in self.active_regions:
                active_points = points.intersection(region.get_halo())
                
                while active_points and region.active:
                    for point in active_points:
                        if point in added_points:
                            #regions meet, we found a saddle.
                            region2 = added_points[point]
                            region.set_saddle(point)
                            region2.set_saddle(point)

                            halos = region.halo.union(region2.halo)
                            edges = region.edge.union(region2.edge)
                            borders = halos.union(edges)
                            
                            #TODO: components returns list of connected
                            #components, largest one in front
                            #comp = components(borders)
                            comp = [borders]#TODO remove this!
                            
                            region2.halo = comp[0].intersection(halos)
                            
                            if len(comp)>1:
                                region.halo = comp[1].intersection(halos)
                            else:
                                region.passivate()
                                
                            for c in comp[2:]:
                                pass
                                #TODO make new region and grow it
                                #new_reg.halo.difference_update(edges)
                        else:
                            region.add(point)
                            added_points[point] = region        
                            #TODO test for plateau saddle:
                            #test halo locally for connectedness
                            #if disconn. test globally with A* going along iso surfaces
                            #if really disconn, mark saddle and choose one component as new halo

                    active_points = points.intersection(region.get_halo())
                    
                if not region.active:
                    self.passive_regions.append(region)
                    self.active_regions.remove(region)

            #then look at remaining points and create new regions as necessary
            new_regions = []
            remaining_points = points.difference(added_points)
            while remaining_points:
                point = remaining_points.pop()
                region = Region(point, self.array)
                added_points[point] = region
                new_regions.append(region)

                #now fill the new region as much as possible
                active_points = remaining_points.intersection(region.get_halo())
                while active_points:
                    for point in active_points:
                        #TODO test for special points and act accordingly
                        #can only plateau saddles happen here?
                        region.add(point)
                        added_points[point] = region
                    active_points = remaining_points.intersection(region.get_halo())
                    
                #and update which points are left now
                remaining_points = points.difference(added_points)
                
            self.active_regions += new_regions
    ##        if level > 115 and level < 125:
    ##            plot_debug(active_regions + passive_regions, a)
    ##            region.plot()
        self.regions = self.active_regions + self.passive_regions

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        return "Decompostition of a "+str(self.array.shape)+\
               " "+str(self.array.dtype)+" array into "+\
               str(len(self))+" slope regions."
                
    #plot 2D array in 3D projection
    def plot(self):
        assert self.array.ndim==2
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.tight_layout()
        plt.subplots_adjust(top = 1,
                            bottom = 0,
                            right = 1,
                            left = 0,
                            hspace = 0,
                            wspace = 0)

        #plot original data as wire
        x = range(self.array.shape[0])
        y = range(self.array.shape[1])
        x, y = np.meshgrid(x, y)
        ax.plot_wireframe(x.T, y.T, self.array,
                          colors="k",
                          linewidths=0.2)

        #plot colorful markers to indicate different regions
        colors = ["b", "r", "k", "c", "g", "y", "m"]
        markers = ["o", "x", "s", "+", "*"]
        for k, region in enumerate(self.regions):
            xs = [p[0] for p in region.points]
            ys = [p[1] for p in region.points]
            zs = [self.array[p] for p in region.points]
            ax.scatter(xs, ys, zs,
                       zdir="z",
                       zorder=1,
                       s=35,
                       c=colors[k%7],
                       marker=markers[k%5],
                       depthshade=False)
        plt.show()

#silly function only for debug
def plot_debug(regions, a):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.tight_layout()
    plt.subplots_adjust(top = 1,
                        bottom = 0,
                        right = 1,
                        left = 0,
                        hspace = 0,
                        wspace = 0)

    #plot original data as wire
    x = range(a.shape[0])
    y = range(a.shape[1])
    x, y = np.meshgrid(x, y)
    ax.plot_wireframe(x.T, y.T, a,
                      colors="k",
                      linewidths=0.2)

    #plot colorful markers to indicate different regions
    colors = ["b", "r", "k", "c", "g", "y", "m"]
    markers = ["o", "x", "s", "+", "*", "."]
    for k, region in enumerate(regions):
        xs = [p[0] for p in region.points]
        ys = [p[1] for p in region.points]
        zs = [a[p] for p in region.points]
        ax.scatter(xs, ys, zs,
                   zdir="z",
                   zorder=1,
                   s=10,
                   c=colors[k%7],
                   marker=markers[k%5],
                   depthshade=False)
    ax.view_init(elev=40, azim=150)
    plt.show()


if __name__ == "__main__":
    #dummy data for debug
    #d = np.round(10*np.random.rand(6,6)).astype(np.int)
    pic = Image.open("mediumTestImage.png")
    data = np.array(pic)[..., 1]
    d=SlopeDecomposition(data)
    d.plot()
