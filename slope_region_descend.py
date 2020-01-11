import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from copy import copy
from tqdm import tqdm
#from numba import jit, njit
import itertools

class Region:

    def __init__(self, point, decomp):
        self.decomp = decomp
#        self.min_idx = point
#        self.max_idx = None
#        self.sad_idx = set() #needn't be a point in the region!
        self.active = True
        self.points = set() #points, both inner and on the edge
#        self.edge = set()   #border, belonging to the region
        self.halo = set()   #border, not (yet) part of the region
        self.add(point)
        self.id = len(decomp.regions)

    def __repr__(self):
        return str(self.points)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        yield self.points
        
    def passivate(self):
        self.active = False
        self.halo = set()

    def add(self, point):
        assert point not in self.points
        self.points.add(point)
        self.decomp.unassigned_points.remove(point)

        neigh = self.decomp.get_neigh(point)
        new_halo = neigh.difference(self.points)

        self.halo.update(new_halo)
        self.halo.discard(point)

#        self.edge.add(point)
#        for ned in neigh.intersection(self.edge):
#            if self.decomp.get_neigh(ned).issubset(self.points):
#                self.edge.remove(ned)

#        #TODO this is wrong
#        if not new_halo:
#            #we found a maximum
#            self.max_idx = point
##            self.edge.difference_update(neigh)
#            #self.passivate()




class SlopeDecomposition:

    @property
    def regions(self):
        return self.active_regions + self.passive_regions

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        return "Decompostition of a "+str(self.array.shape)+\
               " "+str(self.array.dtype)+" array into "+\
               str(len(self))+" slope regions."

    def __init__(self, array):
        assert array.ndim > 1

        self.array = array

        # for use in get_cube
        #self.shape_m_1 = np.array(array.shape) - 1
        self.dim = len(array.shape)
        self.offset_array = itertools.product(range(-1,2), repeat = self.dim)
        self.offset_array = np.array(list(self.offset_array))

        self.active_regions = []
        self.passive_regions = []

        # indices of points not yet assigned to any region
        self.unassigned_points = {tuple(idx) for idx in np.array(np.unravel_index(
                                    range(array.size), array.shape)).T}

        # sort indices for increasing array value
        sorted_idx = np.unravel_index(array.argsort(axis=None),
                                      array.shape)

        # create empty levelsets for each value that occurs
        self.levelsets = {val : set() for val in array[sorted_idx]}

        # then fill in the indices
        for idx in np.array(sorted_idx).T:
            self.levelsets[array[tuple(idx)]].add(tuple(idx))

        # then sort by level
        self.levelsets_sorted = sorted(self.levelsets.items(), key = lambda x:x[0])

        #self.decompose()


    def decompose(self):
        for lvl, points in (self.levelsets_sorted):
            self.decomposeStep(lvl, points)


    def doDecomposeStep(self):
        for lvl, points in (self.levelsets_sorted):
            self.decomposeStep(lvl, points)
            yield lvl


    def decomposeStep(self, lvl, points):

        # first off, deal with points that can be assigned to existing regions
        while any([r.halo.intersection(points, self.unassigned_points) for r in self.active_regions]):
            for region in self.active_regions:
                active_points = points.intersection(region.halo, self.unassigned_points)

                while active_points:
                    point = active_points.pop()
                    region.add(point)
                    #self.unassigned_points.remove(point) #in region.add() now!

                    # test local connectedness around point as fast heuristic
                    local_env = self.get_cube(point).intersection(self.unassigned_points)

                    if local_env: #TODO: is there really nothing to do otherwise?

                        if len(self.find_connected_components(local_env, local_env)) > 1:
                            # test global connectedness now
                            #TODO: look at these cases, whether we can do another cheap test
                            components = self.find_connected_components(local_env, self.unassigned_points)
                            
                            # sort components, biggest chunk of unassigned points in front
                            components = sorted(components, key = len, reverse=True)
                        else:
                            components = [self.unassigned_points]

                        # list of all the regions with point in their halo
                        involved_regions = [region] + [r for r in self.active_regions
                                                       if point in r.halo]

                        # union of halos of involved regions
                        involved_halos = set()
                        for r in involved_regions:
                            involved_halos.update(r.halo)
                        involved_halos.intersection_update(self.unassigned_points)

                        halo_components = [involved_halos.intersection(c) for c in components]

                        compo_and_halo = list(zip(components, halo_components))

                        # now distribute halo components to the involved regions
                        for r in involved_regions:
                            # remember the current halo and wipe it
                            found_halo = False
                            old_halo = r.halo
                            r.halo = set()

                            # then assign a halo component to the region
                            # the "copy" here is needed for correct iteration
                            for c, h in copy(compo_and_halo):
                                if old_halo.intersection(h):
                                    # this halo component is on the border of r,
                                    # and can therefor be assigned to r.
                                    
                                    is_plateau = c.issubset(points)
                                    if is_plateau or not found_halo:
                                        # we can add the current component, if
                                        # it is a plateau or if we haven't found
                                        # a halo for the region yet.
                                        
                                        r.halo.update(h)
                                        compo_and_halo.remove((c,h))
                                        if not is_plateau:
                                            # in this case we just found a halo
                                            found_halo = True

                            if not found_halo:
                                r.passivate()

                        # deal with remaining components
                        while compo_and_halo:
                            # grab a point from a remaining halo,
                            # and start a region from there.
                            c,h = compo_and_halo.pop()
                            point = h.pop()
                            r = Region(point, self)
                            #self.unassigned_points.remove(point)
                            self.active_regions.append(r)

                            # add the whole halo component to the region
                            for p in h:
                                r.add(p)
                                #self.unassigned_points.remove(p)
                            r.halo.intersection_update(self.unassigned_points)

                            # now look at the connected components of
                            # the region halo in the unassigned points.
                            # if we get more than one component, there
                            # was a self-collision and we need to create
                            # additional regions for the new components.
                            # if the halo is empty however, we don't have to do anything.
                            if r.halo:
                                h_compo = self.find_connected_components(r.halo, self.unassigned_points)
                                compo_and_halo += [(c, r.halo.intersection(c)) for c in h_compo[1:]]
                                r.halo.intersection_update(h_compo[0])

                    active_points = points.intersection(region.halo, self.unassigned_points)

                if not region.active:
                    self.passive_regions.append(region)
                    self.active_regions.remove(region)

        #then look at remaining points and create new regions as necessary
        new_regions = []
        remaining_points = points.intersection(self.unassigned_points)
        while remaining_points:
            point = remaining_points.pop()
            region = Region(point, self)
            #self.unassigned_points.remove(point)
            new_regions.append(region)

            #now fill the new region as much as possible
            active_points = remaining_points.intersection(region.halo)
            while active_points:
                for point in active_points:
                    #TODO test for self-collision
                    region.add(point)
                    #self.unassigned_points.remove(point)
                active_points = remaining_points.intersection(region.halo)

            #and update which points are left now
            remaining_points = points.intersection(self.unassigned_points)

        self.active_regions += new_regions

        # cut away parts of the halos which are already assigned
        for region in self.regions:
            region.halo.intersection_update(self.unassigned_points)


    def find_connected_components(self, small_set, big_set):
        assert small_set.issubset(big_set)
        assert len(small_set) != 0

        small_set_copy = copy(small_set)

        components = []
        while small_set_copy:
            seed = small_set_copy.pop()
            border = {seed}
            component = set()
            while border:
                point = border.pop()
                component.add(point)
                border.update(self.get_cube(point).intersection(big_set).difference(component))
            components.append(component)
            small_set_copy.difference_update(component)
        return components


    #get points with index varying at most by 1 in every dimension
    #attention when changing: same code duplicated in Region
    #@njit
    def get_cube(self, point):
        idx = np.array(point)

        # check whether we have a point on the
        # edge of the array or an inner point
#        inner_point = not 0 in point
#        for n,i in enumerate(point):
#            inner_point *= i<self.array.shape[n]

#        inner_point = not(np.any(point == 0) or np.any(point == self.shape_m_1))

#        if True or inner_point :
            # use precalculated offsets
            # TODO
        return {tuple(p) for p in self.offset_array + idx}
#        else:
##            low_corner = idx-1
##            low_corner[low_corner<0] = 0
#            low_corner = np.maximum(idx - 1, 0)
#            high_corner = np.minimum(idx + 1, self.shape_m_1)
#            offsets = [np.array(i)
#                       for i in np.ndindex(tuple(high_corner+1-low_corner))]
#
#            return {tuple(low_corner+o) for o in offsets}
#
    #get points that are directly adjacent along the axes
    #attention when changing: same code duplicated in Region
    def get_neigh(self, point):
        neigh = set()
        for dim, len in enumerate(self.array.shape):
            pos = point[dim]
            for k in [max(0, pos-1), pos, min(len-1, pos+1)]:
                new_idx = tuple(p if i!=dim else k for i,p in enumerate(point))
                neigh.add(new_idx)
        return neigh



if __name__ == "__main__":


    profiling_mode = False

    #dummy data for debug
    #d = np.round(10*np.random.rand(6,6)).astype(np.int)
#    pic = Image.open("brain.png")
    pic = Image.open("perlin_small.png")
#    pic = Image.open("mediumTestImage.png")
    data = np.array(pic)[..., 1]
#    data = 255-data
    d=SlopeDecomposition(data)
    #d.plot()

    if profiling_mode:
        d.decompose()
    else:

        steps = 0

        gen = d.doDecomposeStep()
        def step():
            try:
                gen.__next__()
            except StopIteration:
                pass



        alpha = 128

        colors = ((0xff, 0x9f, 0x1c, alpha),
                  (0xad, 0x34, 0x3e, alpha),
                  (0x06, 0x7b, 0xc2, alpha),
                  (0xd3, 0x0c, 0xfa, alpha),
                  (0x0c, 0xfa, 0xfa, alpha),
                  (0x18, 0xe7, 0x2e, alpha),
                  (0x23, 0x09, 0x03, alpha),
                  (0xdb, 0x54, 0x61, alpha),
                  (0x19, 0x72, 0x78, alpha),
                  (0xee, 0x6c, 0x4d, alpha))


        import pygame

        print_debug_colors = False

        pixelsize = 5

        pygame.init()
        pygame.display.set_caption("Border Propagation")
        clock = pygame.time.Clock()

        def screeninit():
            global screen, region_surface, bordersize

            bordersize = pixelsize//3
            screensize = (pixelsize * data.shape[0], pixelsize * data.shape[1])

            screen = pygame.display.set_mode(screensize)

            region_surface = pygame.Surface(screensize, flags = pygame.SRCALPHA)
            region_surface.set_alpha(alpha)

        screeninit()

        try:

            done = False
            while (not done):
                # --- Main event loop
                for event in pygame.event.get(): # User did something
                    if event.type == pygame.QUIT: # If user clicked close
                        done = True # Flag that we are done so we exit this loop
                    if event.type == pygame.KEYDOWN:
                        if event.key in [pygame.K_ESCAPE, pygame.K_BACKSPACE, pygame.K_F4]:
                            done = True
                        elif event.key == pygame.K_SPACE:
                            steps += 1
                        elif event.key == pygame.K_RIGHT:
                            steps += 10
                        elif event.key == pygame.K_UP:
                            steps += 256
                        elif event.key == pygame.K_DOWN:
                            steps = 0
                        elif event.key == pygame.K_F5:
                            print_debug_colors = not print_debug_colors
                        elif event.key == pygame.K_KP_PLUS:
                            pixelsize += 1
                            screeninit()
                        elif event.key == pygame.K_KP_MINUS:
                            pixelsize -= 1 * (pixelsize > 1)
                            screeninit()

                if steps > 0:
                    steps -= 1
                    step()


                # 1. Draw data
                for i in range(data.shape[0]):
                    for j, v in enumerate(data[i]):
                        screen.fill((v,v,v), rect = (pixelsize*i,
                                                     pixelsize*j,
                                                     pixelsize,
                                                     pixelsize))

                # 2. Draw Regions
                for r in d.regions:

                    region_surface.fill((0,0,0,0))

                    for p in r.points:
                        region_surface.fill(colors[r.id%len(colors)], rect = (pixelsize*p[0],
                                                                   pixelsize*p[1],
                                                                   pixelsize,
                                                                   pixelsize))

                    # 3. Draw Halos
                    for p in r.halo:
                        region_surface.fill(colors[r.id%len(colors)], rect = (pixelsize*p[0] + bordersize,
                                                                   pixelsize*p[1] + bordersize,
                                                                   pixelsize - 2*bordersize,
                                                                   pixelsize - 2*bordersize))

                    screen.blit(region_surface, (0,0))

                # 4. Draw current point


                # 5. Draw all colors for debugging.
                if print_debug_colors:
                    for i, c in enumerate(colors):
                        screen.fill(c, rect = (2*pixelsize*i,
                                               0,
                                               2*pixelsize,
                                               2*pixelsize))






                pygame.display.flip()

                clock.tick(60)

        finally:
            pygame.quit()



        # Note: Reeb-Graphs
        # Countour Tree
        # Maximally Stable Extremal Refions
        # Watershed

        # create monkey_saddle.png





