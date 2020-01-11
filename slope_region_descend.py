import itertools
import numpy as np
from PIL import Image
from copy import copy
#from tqdm import tqdm

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
        new_halo = neigh - self.points
        new_halo &= self.decomp.unassigned_points

        self.halo |= new_halo
        self.halo.discard(point)



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
        
        # for use in get_neigh
        self.a_shape = array.shape

        # for use in get_cube
        self.dim = len(array.shape)
        self.offset_array = itertools.product(range(-1,2), repeat = self.dim)
        self.offset_array = np.array(list(self.offset_array))
        
        # keep track of active and passive regions
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

    # get points with index varying at most by 1 in every dimension
    # this might produce out-of-bounds indices, beware!
    def get_cube(self, point):
        return {tuple(p) for p in self.offset_array + np.array(point)}

    # get points that are directly adjacent along the axes
    def get_neigh(self, point):
        neigh = set()
        for dim, len in enumerate(self.a_shape):
            pos = point[dim]
            for k in [max(0, pos-1), pos, min(len-1, pos+1)]:
                new_idx = tuple(p if i!=dim else k
                                for i,p in enumerate(point))
                neigh.add(new_idx)
        return neigh

    def decompose(self):
        for lvl, points in (self.levelsets_sorted):
            self.decomposeStep(lvl, points)

    def doDecomposeStep(self):
        for lvl, points in (self.levelsets_sorted):
            self.decomposeStep(lvl, points)
            yield lvl

    def decomposeStep(self, lvl, points):
        # first off, deal with points that can be assigned to existing regions
        while any([r.halo & points for r in self.active_regions]):
            # make sure the halos consist only of unassigned points
            for r in self.regions:
                r.halo &= self.unassigned_points
                
            for region in self.active_regions:
                active_points = points & region.halo

                while active_points:
                    point = active_points.pop()
                    region.add(point)

                    # test local connectedness around point as fast heuristic
                    local_env = self.get_cube(point) & self.unassigned_points

                    if local_env: #TODO: is there really nothing to do otherwise?

                        if len(self.find_connected_components(local_env, local_env)) > 1:
                            # test global connectedness now
                            #TODO: look at these cases, whether we can do another cheap test.
                            #perhaps a larger local region?
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
                            involved_halos |= r.halo

                        halo_components = [involved_halos & c for c in components]

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
                                if old_halo & h:
                                    # this halo component is on the border of r,
                                    # and can therefor be assigned to r.
                                    
                                    is_plateau = c<=points
                                    if is_plateau or not found_halo:
                                        # we can add the current component, if
                                        # it is a plateau or if we haven't found
                                        # a halo for the region yet.
                                        
                                        r.halo |= h
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
                            self.active_regions.append(r)

                            # add the whole halo component to the region
                            for p in h:
                                r.add(p)

                            # now look at the connected components of
                            # the region halo in the unassigned points.
                            # if we get more than one component, there
                            # was a self-collision and we need to create
                            # additional regions for the new components.
                            # if the halo is empty however, we don't have to do anything.
                            if r.halo:
                                h_compo = self.find_connected_components(r.halo, self.unassigned_points)
                                compo_and_halo += [(c, r.halo & c) for c in h_compo[1:]]
                                r.halo &= h_compo[0]

                    active_points = points & region.halo

                if not region.active:
                    self.passive_regions.append(region)
                    self.active_regions.remove(region)

        # then look at remaining points and create new regions as necessary
        #TODO: think of a better way to deal with remaining points
        #we want to do similar stuff to the "while compo_and_halo" loop above.
        new_regions = []
        remaining_points = points.intersection(self.unassigned_points)
        while remaining_points:
            point = remaining_points.pop()
            region = Region(point, self)
            new_regions.append(region)

            # now fill the new region as much as possible
            active_points = remaining_points.intersection(region.halo)
            while active_points:
                for point in active_points:
                    #TODO test for self-collision
                    region.add(point)
                active_points = remaining_points.intersection(region.halo)

            # and update which points are left now
            remaining_points = points.intersection(self.unassigned_points)

        self.active_regions += new_regions


    def find_connected_components(self, small_set, big_set):
        assert small_set <= big_set
        assert small_set

        small_set_copy = copy(small_set)

        components = []
        while small_set_copy:
            seed = small_set_copy.pop()
            border = {seed}
            component = set()
            while border:
                #TODO: implement A* search and break as soon as there's only one component
                point = border.pop()
                component.add(point)
                border |= (self.get_cube(point) & big_set) - component
            components.append(component)
            small_set_copy -= component
        return components



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
            #TODO 3D render
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
        # Maximally Stable Extremal Regions
        # Watershed

        # create monkey_saddle.png





