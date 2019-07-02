import autograd.numpy as np
from autograd.extend import primitive, defvjp

from ceviche.utils import circ2eps, grid_coords
from ceviche.primitives import vjp_maker_num

class Param_Base(object):

    def __init__(self):
        pass

    @classmethod
    def get_eps(cls, params, *args):
        raise NotImplementedError("Need to implement a function for computing permittivity from parameters")

""" Topology / Continuous Optimization """

class Param_Topology(Param_Base):

    def __init__(self):
        super().__init__(self)

    @staticmethod
    def _density2eps(mat_density, eps_max):
        return 1 + (eps_max - 1) * mat_density

    @classmethod
    def get_eps(cls, params, eps_background, design_region, eps_max):

        mat_density = params
        eps_inner = cls._density2eps(mat_density, eps_max) * (design_region == 1)
        eps_outer = eps_background * (design_region == 0)

        return eps_inner + eps_outer

""" Shape Optimization """

class Param_Shape(Param_Base):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_eps(*args):
        raise NotImplementedError("Need to implement a function for computing permittivity from parameters")

    @staticmethod
    def sigmoid(x, strength=.02):
        # used to anti-alias the circle, higher strength means sharper boundary
        return np.exp(x * strength) / (1 + np.exp(x * strength))


class Circle_Shapes(Param_Shape):

    def __init__(self, eps_background, dL):
        self.eps_background = eps_background
        self.dL = dL
        self.xs, self.ys = grid_coords(self.eps_background, self.dL)
        super().__init__()

    def circle(self, xs, ys, x, y, r):
        # defines an anti aliased circle    
        dist_from_edge = (xs - x)**2 + (ys - y)**2 - r**2
        return self.sigmoid(-dist_from_edge / self.dL**2)

    def write_circle(self, x, y, r, value):
        # creates an array that, when added to the background epsilon, adds a circle with (x,y,r,value)
        circle_mask = self.circle(self.xs, self.ys, x, y, r)
        new_array = (value - self.eps_background) * circle_mask
        return new_array

    def get_eps(self, xs, ys, rs, values):
        # returns the permittivity array for a bunch of holes at positions (xs, ys) with radii rs and epsilon values (val)
        eps_r = self.eps_background.copy()
        for x, y, r, value in zip(xs, ys, rs, values):
            circle_eps = self.write_circle(x, y, r, value) 
            eps_r += circle_eps
        return eps_r

""" Level Set Optimization """

class Param_LevelSet(Param_Base):

    def __init__(self):
        super().__init__(self)

    @classmethod
    def get_eps(cls, params, eps_background, design_region, eps_max):
        raise NotImplementedError("Need to implement a function for computing permittivity from parameters")


if __name__ == '__main__':

    # example of calling one

    shape = (10, 20)
    params = np.random.random(shape)
    eps_background = np.ones(shape)
    design_region = np.zeros(shape)
    design_region[5:,:] = 1
    eps_max = 5

    eps = Param_Topology.get_eps(params, eps_background, design_region, eps_max)
    
