import autograd.numpy as np
from ceviche.primitives import circ2eps_ag

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
        super().__init__(self)

    @classmethod
    def get_eps(cls, *args):
        raise NotImplementedError("Need to implement a function for computing permittivity from parameters")


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

        eps_r = circ2eps_ag(*args)

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
    
