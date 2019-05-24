import autograd.numpy as np
from autotrack.interpolations import interp

class electron():
    """ electron propagation and initialization """

    def __init__(self, beta):

        # starting conditions
        self.y = np.random.random()   # make gaussian 
        self.x = 0.0                   
        self.px = beta
        self.py = np.random.random()  # make gaussian 
        self.coords = (self.x, self.y, self.px, self.py)

    def update(self, fields):
        """ update with lorentz force """

        self.Fx = interp(self.coords, fields.Ex)  #need to implement
        self.Fy = interp(self.coords, fields.Ey)

    @staticmethod
    def magnetic_force(coords, Hz):
        raise NotImplementedError()
