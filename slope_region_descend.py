import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from copy import copy
from tqdm import tqdm
#from numba import jit, njit

class Region:

    def __init__(self, point, decomp):
        self.decomp = decomp
        self.min_idx = point
        self.max_idx = None
        self.sad_idx = set() #needn't be a point in the region!
        self.active = True
        self.points = set() #points, both inner and on the edge
        self.edge = set()   #border, belonging to the region
        self.halo = set()   #border, not (yet) part of the region
        self.add(point)
        self.id = len(decomp.regions)

    def __repr__(self):
        return str(self.points)

    def __len__(self):
        return len(self.points)

    def __iter__(self):
        yield self.points

    def add(self, point):
        assert point not in self.points
        
        neigh = self.decomp.get_neigh(point)
        self.points.add(point)

        new_halo = neigh.difference(self.points)

        self.halo.update(new_halo)
        self.halo.discard(point)
        
        self.edge.add(point)
        for ned in neigh.intersection(self.edge):
            if self.decomp.get_neigh(ned).issubset(self.points):
                self.edge.remove(ned)

        #TODO this is wrong
        if not new_halo:
            #we found a maximum
            self.max_idx = point
            self.edge.difference_update(neigh)
            #self.passivate()
        
        
    def passivate(self):
        self.active = False
        self.halo = set()


class SlopeDecomposition:

    def __init__(self, array):
        assert array.ndim > 1
        
        self.array = array
        self.active_regions = []
        self.passive_regions = []
        
        #sort indices for increasing array value
        sorted_idx = np.unravel_index(self.array.argsort(axis=None),
                                      self.array.shape)
        
        self.unassigned_points = {tuple(idx) for idx in np.array(np.unravel_index(
                                    range(self.array.size), self.array.shape)).T}
        
        #create empty levelsets for each value that occurs
        self.levelsets = {val : set() for val in self.array[sorted_idx]}

        #then fill in the indices
        for idx in np.array(sorted_idx).T:
            self.levelsets[self.array[tuple(idx)]].add(tuple(idx))
        
        #then sort by level
        self.levelsets_sorted = sorted(self.levelsets.items(), key = lambda x:x[0])
        
        #self.decompose()    
    
    @property
    def regions(self):
        return self.active_regions + self.passive_regions

    def __len__(self):
        return len(self.regions)

    def __repr__(self):
        return "Decompostition of a "+str(self.array.shape)+\
               " "+str(self.array.dtype)+" array into "+\
               str(len(self))+" slope regions."
        
    def decompose(self):
        
#        debug_lvl_stop = 0
        
        for lvl, points in (self.levelsets_sorted):
            
#            print(f"""
#            Doing level: {lvl}
#            Active Regions: {len(self.active_regions)}
#            Points in Lvlset: {len(points)}   
#            """)
#            
#            if lvl >= debug_lvl_stop:
#                i = input()
#                if i == "stop":
#                    return
#                elif i.isdecimal():
#                    debug_lvl_stop = int(i)
#                elif i == "plot":
#                    self.plot()
            
            if lvl > 130:
                break
            
            self.decomposeStep(lvl, points)
            
    
    def doDecomposeStep(self):
        for lvl, points in (self.levelsets_sorted):
            self.decomposeStep(lvl, points)
            yield lvl 
    
    def decomposeStep(self, lvl, points):
  
        #first off, deal with points that can be assigned to existing regions
        while any([r.halo.intersection(points, self.unassigned_points) for r in self.active_regions]):
            for region in self.active_regions:
                active_points = points.intersection(region.halo, self.unassigned_points)
                
                while active_points:
                    point = active_points.pop()
                    region.add(point)
                    self.unassigned_points.remove(point)
                    
                    # test local connectedness around point as fast heuristic
                    local_env = self.get_cube(point).intersection(self.unassigned_points)
                    
                    if local_env:
                    
                        if len(self.find_connected_components(local_env, local_env)) > 1:
                            # test global connectedness now
                            print("beep boop")
                            components = self.find_connected_components(local_env, self.unassigned_points)
                            components = sorted(components, key = len, reverse=True)
                        else:
                            components = [self.unassigned_points]

                        involved_regions = [region] + [r for r in self.active_regions
                                                       if point in r.halo]

                        involved_halos = set()
                        for r in involved_regions:
                            involved_halos.update(r.halo)
                        involved_halos.intersection_update(self.unassigned_points)

                        halo_components = [involved_halos.intersection(c) for c in components]

                        compo_and_halo = list(zip(components, halo_components))

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
                                        # add plateaus to the region halo without
                                        # counting them as a found_halo
                                        r.halo.update(h)
                                        compo_and_halo.remove((c,h))
                                        if not is_plateau:
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
                            self.unassigned_points.remove(point)
                            self.active_regions.append(r)

                            # add the whole halo component to the region
                            for p in h:
                                r.add(p)
                                self.unassigned_points.remove(p)
                            r.halo.intersection_update(self.unassigned_points)

                            # now look at the connected components of
                            # the region halo in the unassigned points.
                            # if we get more than one component, there
                            # was a self-collision and we need to create
                            # additional regions for the new components
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
            self.unassigned_points.remove(point)
            new_regions.append(region)

            #now fill the new region as much as possible
            active_points = remaining_points.intersection(region.halo)
            while active_points:
                for point in active_points:
                    #TODO test for self-collision
                    region.add(point)
                    self.unassigned_points.remove(point)
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
        low_corner = idx-1

        # check whether we have a point on the
        # edge of the array or an inner point
        inner_point = not 0 in point
        for n,i in enumerate(point):
            inner_point *= i<self.array.shape[n]

        if inner_point:
            # use precalculated offsets
            TODO
        else:
            low_corner[low_corner<0] = 0
            high_corner = np.minimum(idx+1, np.array(self.array.shape)-1)
            offsets = [np.array(i)
                       for i in np.ndindex(tuple(high_corner+1-low_corner))]
            
        return {tuple(low_corner+o) for o in offsets}
    
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
    
    
    #dummy data for debug
    #d = np.round(10*np.random.rand(6,6)).astype(np.int)
    pic = Image.open("perlin.png")
    data = np.array(pic)[..., 1]
    d=SlopeDecomposition(data)
    #d.plot()
    #d.decompose()
    
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
    
    pixelsize = 1
    bordersize = 0
    screensize = (pixelsize * data.shape[0], pixelsize * data.shape[1])
    
    
    
    pygame.init()
    screen = pygame.display.set_mode(screensize)
    pygame.display.set_caption("Border Propagation")
    clock = pygame.time.Clock()
    
    region_surface = pygame.Surface(screensize, flags = pygame.SRCALPHA)
    region_surface.set_alpha(alpha)
       
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
                        step()
                    elif event.key == pygame.K_RIGHT:
                        for _ in tqdm(range(10)):
                            step()
                    elif event.key == pygame.K_UP:
                        for _ in tqdm(range(100)):
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
            for i, c in enumerate(colors):
                screen.fill(c, rect = (2*pixelsize*i,
                                       0,
                                       2*pixelsize,
                                       2*pixelsize))
            
            
            
            
            
            
            pygame.display.flip()
            
            clock.tick(60)
    
    finally:    
        pygame.quit()
    
    
