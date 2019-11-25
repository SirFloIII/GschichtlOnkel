import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

class Region:

    def __init__(self, point, array):
        self.array_shape = array.shape
        self.min_idx = point
        self.max_idx = None
        self.sad_idx = None
        self.active = True
        self.points = set() #points, both inner and on the edge
        self.edge = set()   #border, belonging to the region
        self.halo = set()   #border, not (yet) part of the region
        self.add(point)

    def __repr__(self):
        return str(self.points)

    def plot(self):
        assert len(self.array_shape)==2
        
        points = np.array(list(self.points)).T
        plt.scatter(points[0], points[1], c="k", marker=".")

        edge = np.array(list(self.edge)).T
        plt.scatter(edge[0], edge[1], c="g")

        halo = np.array(list(self.halo)).T
        plt.scatter(halo[0], halo[1], c="0.5")
        
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

        self.halo.update(new_halo)
        self.halo.discard(point)
        
        self.edge.add(point)
        if not new_halo:
            #we found a maximum
            self.max_idx = point
            self.edge.difference_update(neigh)
            self.passivate()
        
        #TODO test for plateau saddle:
        #test halo locally for connectedness
        #if disconn. test globally with A* going along iso surfaces
        #if really disconn, mark saddle and choose one component as new halo

    def get_edge(self):
        return self.edge

    def get_halo(self):
        return self.halo

    def remove_from_halo(self, points):
        #TODO or scrap if not needed
        pass

    def set_saddle(self, point):
        self.sad_idx = point
        self.passivate()

    def passivate(self):
        self.active = False



def slopify(a):
    """Partition an array of funtion values into regions of monotonic connectedness

    Parameters:
    a : array_like
    n-dimensional array of real, which are understood as "height".

    Returns:
    [regions] : list
    The list of regions that partition the grid. Look at the "Region" class.

    """
    active_regions = []
    passive_regions = []

    #sort indices for increasing array value
    sorted_idx = np.unravel_index(a.argsort(axis=None), a.shape)

    #create empty levelsets for each value that occurs
    levelset = {val : set() for val in a[sorted_idx]}

    #then fill in the indices
    for idx in np.array(sorted_idx).T:
        levelset[a[tuple(idx)]].add(tuple(idx))

    for level, points in levelset.items():
        #remember which points we add to any region
        added_points = set()

        #first off, deal with points that can be assigned to existing regions
        for region in active_regions:
            active_points = points.intersection(region.get_halo())
            
            while active_points and region.active:
                for point in active_points:
                    if point in added_points:
                        #regions meet, we found a saddle. stop this region now.
                        #TODO: this is too simplistic, and a bug.
                        #test halo for connectedness and assign components
                        #to different regions
                        region.set_saddle(point)
                    else:
                        region.add(point)
                        added_points.add(point)

                active_points = points.intersection(region.get_halo())
                
            if not region.active:
                passive_regions.append(region)
                active_regions.remove(region)

        #then look at remaining points and create new regions as necessary
        new_regions = []
        remaining_points = points.difference(added_points)
        while remaining_points:
            point = remaining_points.pop()
            region = Region(point, a)
            added_points.add(point)
            new_regions.append(region)

            #now fill the new region as much as possible
            keep_going = True
            while keep_going:
                active_points = remaining_points.intersection(region.get_halo())
                for point in active_points:
                    #TODO test for special points and act accordingly
                    region.add(point)
                    added_points.add(point)
                keep_going = active_points  #eventually this becomes empty
                
            #and update which points are left now
            remaining_points = points.difference(added_points)
            
        active_regions += new_regions
        plot_debug(active_regions + passive_regions, a)

    return active_regions + passive_regions

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
    markers = ["o", "x", "s", "+", "*"]
    for k, region in enumerate(regions):
        xs = [p[0] for p in region.points]
        ys = [p[1] for p in region.points]
        zs = [a[p] for p in region.points]
        ax.scatter(xs, ys, zs,
                   zdir="z",
                   zorder=1,
                   s=35,
                   c=colors[k%7],
                   marker=markers[k%5],
                   depthshade=False)
    ax.view_init(elev=0., azim=-160)
    plt.show()

#plot 2D array in 3D projection
def plot_slopes(a):
    assert a.ndim==2
    
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
    markers = ["o", "x", "s", "+", "*"]
    for k, region in enumerate(slopify(a)):
        xs = [p[0] for p in region.points]
        ys = [p[1] for p in region.points]
        zs = [a[p] for p in region.points]
        ax.scatter(xs, ys, zs,
                   zdir="z",
                   zorder=1,
                   s=35,
                   c=colors[k%7],
                   marker=markers[k%5],
                   depthshade=False)
    plt.show()

if __name__ == "__main__":
    #dummy data for debug
    #d = np.round(10*np.random.rand(6,6)).astype(np.int)
    pic = Image.open("mediumTestImage.png")
    data = np.array(pic)[..., 1]
    plot_slopes(data)
