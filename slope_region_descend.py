import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Region:

    def __init__(self, point, array):
        self.array_shape = array.shape
        self.min_idx = point
        self.max_idx = None
        self.sad_idx = None
        self.active = True
        self.points = set()#points, both inner and on the edge
        self.edge = set()#border, belonging to the region
        self.halo = set()#border, not (yet) part of the region
        self.add(point)

    def __repr__(self):
        return str(self.points)

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
        for ed in neigh.intersection(self.edge):
            if self.get_neigh(ed).issubset(self.points):
                #we found a maximum
                self.edge.remove(ed)
                self.max_idx = point
                self.passivate()

    def get_edge(self):
        return self.edge

    def get_halo(self):
        return self.halo

    def remove_from_halo(self, points):
        #TODO
        pass

    def set_saddle(self, point):
        self.sad_idx = point
        self.passivate()

    def passivate(self):
        self.active = False


active_regions = []
passive_regions = []

#dummy data for debug
d = np.round(10*np.random.rand(6,6)).astype(np.int)

#sort indices for increasing array value
sorted_idx = np.unravel_index(d.argsort(axis=None), d.shape)

#create empty levelsets for each value that occurs
levelset = {val : set() for val in d[sorted_idx]}

#then fill in the indices
for idx in np.array(sorted_idx).T:
    levelset[d[tuple(idx)]].add(tuple(idx))

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
        region = Region(point, d)
        added_points.add(point)
        new_regions += [region]

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

#plot 2D array in 3D projection
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.subplots_adjust(top = 1,
                    bottom = 0,
                    right = 1,
                    left = 0,
                    hspace = 0,
                    wspace = 0)
plt.tight_layout()
x = range(d.shape[0])
y = range(d.shape[1])
x, y = np.meshgrid(x, x)
ax.plot_wireframe(x.T, y.T, d,
                  colors="k",
                  linewidths=0.2,
                  zorder=1)
colors = ["b", "r", "k", "c", "g", "y", "m"]
markers = ["o", "x", "s", "+", "*"]
for k, region in enumerate(active_regions+passive_regions):
    xs = [p[0] for p in region.points]
    ys = [p[1] for p in region.points]
    zs = [d[p] for p in region.points]
    ax.scatter(xs, ys, zs, zdir="z", zorder=1,
               s=35, c=colors[k%7], marker=markers[k%5], depthshade=False)
plt.show()
