import numpy as np


class Region:
    max_idx = None
    min_idx = None
    sad_idx = None
    points = set()  #inner points
    edge = set()    #border, belonging to the region
    halo = set()    #border, not (yet) part of the region

    def add(self, point):
        #TODO test for min and saddle, handle them, update edge, points & halo
        pass

    def get_edge(self):
        return edge

    def get_halo(self):
        return halo

    def remove_from_halo(self, points):
        #TODO
        pass


active_regions = []
passive_regions = []

#dummy data for debug
d = np.round(10*np.random.rand(4,4)).astype(np.int)

#sort indices for increasing array value
sorted_idx = np.unravel_index(d.argsort(axis=None), d.shape)

#create empty levelsets for each value that occurs
levelsets = {val : set() for val in d[sorted_idx]}

#then fill in the indices
for idx in np.array(sorted_idx).T:
    levelsets[d[tuple(idx)]].add(tuple(idx))

for level in levelsets:
    added_points = set()
    for region in active_regions:
        points_left = True
        while points_left:
            active_points = intersection(region.get_halo(), levelsets[level])
            points_left = active_points==set()
            for point in active_points:
                #TODO test for special points and act accordingly
                region.add(point)
                added_points.add(point)
                active_points.remove(point)

    new_regions = []
    remaining_points = difference(levelsets[level], added_points)
    while not remaining_points==set():
        region = Region()
        new_regions += [region]
        for point in remaining_points:
            #TODO test for special points and act accordingly
            region.add(point)
            added_points.add(point)
        remaining_points = difference(levelsets[level], added_points)
    active_regions += new_regions
