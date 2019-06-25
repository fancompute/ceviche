import autograd.numpy as np

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

        mat_density = params.reshape(eps_background.shape)
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


class Hole_Array(Param_Shape):

    def __init__(self):
        super().__init__(self)

    @classmethod
    def get_eps(cls, *args):
        # implement me!
        pass


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
    
