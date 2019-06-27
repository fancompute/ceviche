import autograd.numpy as np
from autograd.extend import primitive, defvjp

from ceviche.utils import grid_coords
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
        super().__init__()

    @staticmethod
    def _density2eps(mat_density, eps_max):
        return 1 + (eps_max - 1) * mat_density

    def get_eps(self, params, eps_background, design_region, eps_max):

        mat_density = params
        eps_inner = self._density2eps(mat_density, eps_max) * (design_region == 1)
        eps_outer = eps_background * (design_region == 0)

        return eps_inner + eps_outer

""" Shape Optimization """

class Param_Shape(Param_Base):

    def __init__(self, arg_indices=None, step_sizes=None):
        super().__init__()
        self.link_numerical_vjp(arg_indices, step_sizes)

    def get_eps(self, *args):
        raise NotImplementedError("Need to implement a function for computing permittivity from parameters")

    @classmethod
    def link_numerical_vjp(cls, arg_indices, step_sizes):
        vjp_args = vjp_maker_num(cls.get_eps, arg_indices, step_sizes)
        defvjp(cls.get_eps, *vjp_args, None, None)

class Circle_Shapes(Param_Shape):

    def __init__(self, arg_indices=None, step_sizes=None):
        super().__init__(arg_indices=arg_indices, step_sizes=step_sizes)

    @staticmethod
    def hole(x, y, x0, y0, r):
        """ returns True if position x,y is within a hole at center x0,y0 with radius r """
        return (x - x0)**2 + (y - y0)**2 < r**2

    @primitive
    def get_eps(self, xs, ys, rs, eps_holes, eps_background, dL):

        # get coordinates of x and y grid cells
        x_coords, y_coords = grid_coords(eps_background, dL)
        eps_r = eps_background.copy()

        for x, y, r, eps_c in zip(xs, ys, rs, eps_holes):
            within_hole = self.hole(x_coords, y_coords, x, y, r)
            eps_r[within_hole] = eps_c

        return eps_r

# # note, not sure where the best place to do this is, might need to not use classmethods and just initialize parameterization objects instead and do it in __init__.. need to insert dL here
# (dx, dy, dr, deps) = vjp_maker_num(Circle_Shapes.get_eps, list(range(4)), [1e-6, 1e-6, 1e-6, 1e-6])
# defvjp(Circle_Shapes.get_eps, dx, dy, dr, deps, None, None)

""" Old code below 
class Circle_Shapes(Param_Shape):

    def __init__(self):
        super().__init__(self)

    @classmethod
    def get_eps(cls, params, eps_background, dL):
        '''
        Initizlize circles at position (x, y) of radius r and permittivity eps_c
        each defined in the rows of the [4 x Nholes] array 'params'
        '''
        args = []
        args.append(params[0, :]) # x
        args.append(params[1, :]) # y
        args.append(params[2, :]) # r
        args.append(params[3, :]) # eps_c
        args.append(eps_background)
        args.append(dL)

        circ2eps_ag = primitive(circ2eps)
        (dx, dy, dr, deps) = vjp_maker_num(circ2eps, list(range(4)), [dL, dL, dL, 1e-6])

        defvjp(circ2eps_ag, dx, dy, dr, deps, None, None) 

        return circ2eps_ag(*args)
"""

""" Level Set Optimization """

class Param_LevelSet(Param_Base):

    def __init__(self):
        super().__init__(self)

    def get_eps(self, params, eps_background, design_region, eps_max):
        raise NotImplementedError("Need to implement a function for computing permittivity from parameters")


if __name__ == '__main__':

    # example of calling one

    shape = (10, 20)
    params = np.random.random(shape)
    eps_background = np.ones(shape)
    design_region = np.zeros(shape)
    design_region[5:,:] = 1
    eps_max = 5

    eps = Param_Topology().get_eps(params, eps_background, design_region, eps_max)
    
