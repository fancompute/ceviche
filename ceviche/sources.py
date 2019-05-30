import numpy as np
import matplotlib.pylab as plt

""" Defines sources to add to FDTD """

class Source():

    def __init__(self, mask, component):
        """ Make a source
                mask: where the source is applied in grid points, boolean array.
                component: direction and current type (string {'Jx', 'Jy', 'Jz'})
        """

        self._check_inputs_main(mask, component)

    def _check_inputs_main(self, mask, component):
        """ Checks the inputs common to all sources """

        # assert mask.ndim == 3 or mask.ndim == 4, "mask must be 3 or 4 dimensional array"
        # if mask.ndim == 3:
            # mask = mask[:,:,:,None]
        self.mask = 0 * (mask <= 0) + 1 * (mask > 0)

        assert component in ('Jx', 'Jy', 'Jz'), "`component` must be one of {'Jx', 'Jy', 'Jz'}"
        self.component = component

    def time_fn(self):
        raise NotImplementedError("must implement a `time_fn` for this class, which returns the amplitude of the source as a function of time index")

    def J_fn(self, t_index):
        # returns a numpy array corresponding to the J to insert at t_index
        return self.mask * self.time_fn(t_index)

    def plot_fn(self, steps):
        """ Plot the time function for a number of steps """

        times = np.arange(steps)
        amps = self.time_fn(times)

        plt.clf()
        plt.plot(amps)
        plt.xlabel('time steps')
        plt.ylabel('source amplitude')
        plt.show()

    def __repr__(self):
        return 'Source(mask.shape={})'.format(self.mask.shape)

    def __str__(self):
        return "Source with mask of shape {}".format(self.mask.shape)


class Gaussian(Source):

    def __init__(self, mask, component, amp, sigma, t0, freq=None):
        """ adds a gaussian source to the simulation
                mask: position of the source, (3 dimensional array)
                amp: source amplitude (float)
                sigma: pulse duration (time steps)
                t0: time delay (time steps)
        """

        super().__init__(mask, component)
        self._check_inputs(amp, sigma, t0, freq)

    def _check_inputs(self, amp, sigma, t0, freq):
        """ Make sure things are kosher"""

        self.amp = amp
        self.sigma = sigma
        self.t0 = t0
        self.freq = freq

    def _exponential(self, t_index, t0, sigma):
        return np.exp(-(t_index - t0)**2 / 2 / sigma**2)

    def time_fn(self, t_index):
        """ A gaussian function """

        if self.freq is None:
            return self.amp * self._exponential(t_index, self.t0, self.sigma)
        else:
            return self.amp * self._exponential(t_index, self.t0, self.sigma) * np.cos(2 * np.pi * freq * t_index)

    def __repr__(self):
        return "Gaussian(mask.shape={}, component='{}', amp={}, sigma={}, t0={})".format(self.mask.shape, self.component, self.amp, self.sigma, self.t0)

    def __str__(self):
        return self.__repr__()


class CW(Source):

    def __init__(self, mask, component, amp, t0, freq):
        """ adds a CW wave source to the simulation
                mask: position of the source, (3 dimensional array)
                amp: source amplitude (float)
                t0: time delay (time steps)
                freq: frequency (1/time steps)
        """

        super().__init__(mask, component)
        self._check_inputs(amp, t0, freq)

    def _check_inputs(self, amp, t0, freq):
        """ Make sure things are kosher"""

        self.amp = amp
        self.t0 = t0
        self.freq = freq

    def _sigmoid(self, t_index, t0, strength=10):
        """ Simple sigmoid function to model the ramping up
                strength: controls the 'speed' of the ramp.
        """
        return 1 / (1 + np.exp(-(t_index - t0/2) * strength / t0))

    def time_fn(self, t_index):
        """ A gaussian function with optional time periodic modulation """

        return self.amp * np.cos(2 * np.pi * t_index * self.freq) * self._sigmoid(t_index, self.t0)

    def __repr__(self):
        return "CW(mask.shape={}, component='{}', amp={}, t0={})".format(self.mask.shape, self.component, self.amp, self.t0)

    def __str__(self):
        return self.__repr__()
