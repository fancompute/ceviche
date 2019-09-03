# ceviche [![Build Status](https://travis-ci.com/twhughes/ceviche.svg?token=ZCPktA3Ki2eYVXYnfbrz&branch=master)](https://travis-ci.com/twhughes/ceviche)

Electromagnetic Simulation Tools + Automatic Differentiation.  Code for the arxiv preprint [Forward-Mode Differentiation of Maxwell's Equations](https://arxiv.org/abs/1908.10507).

## What is ceviche?

`ceviche` provides two core electromagnetic simulation tools for solving Maxwell's equations:

- finite-difference frequency-domain (FDFD)

- finite-difference time-domain (FDTD)

Both are written in `numpy` / `scipy` and are compatible with the [HIPS autograd package](https://github.com/HIPS/autograd).

This allows you to write code to solve your E&M problem, and then use automatic differentiation on your results.

As a result, you can do gradient-based optimization or sensitivity analysis without the tedius process of deriving your derivatives analytically.

### A simple example

Lets say we have a domain of where we wish to inject light at position `source` and measure its intensity at `probe`.

Between these two points, there's a box at location `pos_box` with permittivity `eps`.

We can write a function computing the intensity as a function of `eps` using our FDFD solver

```python
import autograd.numpy as np           # import the autograd wrapper for numpy
from ceviche import fdfd_ez as fdfd   # import the FDFD solver

# make an FDFD simulation
f = fdfd(omega, dl, eps_box, npml=[10, 10])

def intensity(eps):
    """ computes electric intensity at `probe` for a given box permittivity of `eps`

        source |-----| probe
            .  | eps |  .
               |_____|
    """

    # set the permittivity in the box region to the input argument
    fdfd.eps_r[box_pos] = eps

    # solve the fields
    Ex, Ey, Hz = f.solve(source)

    # compute the intensity at `probe`
    I = np.square(np.abs(Ex)) + np.square(np.abs(Ex))
    return = np.sum(I * probe)
```

Then, we can very easily differentiate this function using automatic differentiation

```python

# use autograd to differentiate `intensity` function
grad_fn = jacobian(intensity)

# then, evaluate it at the current value of `eps`
dI_deps = grad_fn(eps_curr)

# or do gradient based optimization
for _ in range(10):
    eps_current += step_size * dI_deps_fn(eps_current)
```

## Design Principle

`ceviche` is designed with simplicity in mind and is meant to serve as a base package for building your projects from.  However, with some exceptions, it does not provide streamlined interfaces for optimization, source or device creation, or visualization.  If you want that kind of thing, you need to build it around the base functionality of ceviche in your own project.  This decision was made to keep things clean and easy to understand, with a focus on the meaty bits that make this package unique.  For some inspiration, see the `examples` directory.  


For more user friendly features, check out our [`angler`](https://github.com/fancompute/angler) package.  We plan to merge the two packages at a later date to give these automatic differentiation capabilities to `angler`.

## Installation

`ceviche` is not on PyPI yet.
To install locally from source:

    git clone https://github.com/twhughes/ceviche.git
    pip install -e ceviche
    pip install -r ceviche/requirements.txt

from the main directory.

Alternatively, just import the package from within your python script

    import sys
    sys.path.append('path/to/ceviche')

## Package Structure

### Ceviche

The `ceviche` directory contains everything needed.

To get the FDFD and FDTD simulators, import directly `from ceviche import fdtd, fdfd_ez, fdfd_hz, fdfd_ez_nl`

To get the differentiation, import `from ceviche import jacobian`.

`constants.py` contains some constants `EPSILON_0`, `C_0`, `ETA_0`, `Q_E`, which are needed throughout the package

`utils.py` contains a few useful functions for plotting, autogradding, and various other things.

### Examples

There are many demos in the `examples` directory, which will give you a good sense of how to use the package.

### Tests

Tests are located in `tests`.  To run, `cd` into `tests` and

    python -m unittest

to run all or

    python specific_test.py

to run a specific one.  Some of these tests involve visual inspection of the field plots rather than error checking on values.

To run all of the gradient checking functions, run 

    bash tests/test_all_gradients.sh

## Citation

If you use this for your research or work, please cite

    @misc{1908.10507,
    Author = {Tyler W Hughes and Ian A D Williamson and Momchil Minkov and Shanhui Fan},
    Title = {Forward-Mode Differentiation of Maxwell's Equations},
    Year = {2019},
    Eprint = {arXiv:1908.10507},
    }
