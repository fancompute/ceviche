# ceviche [![Build Status](https://travis-ci.com/twhughes/ceviche.svg?token=ZCPktA3Ki2eYVXYnfbrz&branch=master)](https://travis-ci.com/twhughes/ceviche)

Electromagnetic Simulation Tools + Automatic Differentiation.  Code for the arxiv preprint [Forward-Mode Differentiation of Maxwell's Equations](https://arxiv.org/abs/1908.10507).

<img src="/img/horizontal-color.png" title="ceviche" alt="ceviche">

(logo by [@nagilmer](http://nadinegilmer.com/))

## What is ceviche?

`ceviche` provides two core electromagnetic simulation tools for solving Maxwell's equations:

- finite-difference frequency-domain (FDFD)

- finite-difference time-domain (FDTD)

Both are written in `numpy` / `scipy` and are compatible with the [HIPS autograd package](https://github.com/HIPS/autograd), supporting forward-mode and reverse-mode automatic differentiation.

This allows you to write code to solve your E&M problem, and then use automatic differentiation on your results.

As a result, you can do gradient-based optimization, sensitivity analysis, or plug your E&M solver into a machine learning model without the tedius process of deriving your derivatives analytically.

### Tutorials

There is a comprehensive ceviche tutorial available at [this link](https://github.com/fancompute/workshop-invdesign) with several ipython notebook examples:
1. [Running FDFD simulations in ceviche.](https://github.com/fancompute/workshop-invdesign/blob/master/01_First_simulation.ipynb)
2. [Performing inverse design of a mode converter.](https://github.com/fancompute/workshop-invdesign/blob/master/02_Invdes_intro.ipynb)
3. [Adding fabrication constraints and device parameterizations.](https://github.com/fancompute/workshop-invdesign/blob/master/03_Invdes_parameterization.ipynb)
4. [Inverse design of a wavelength-division multiplexer and advanced topics.](https://github.com/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb)

There are also a few examples in the `examples/*` directory.

### What can it do?  An Example

Let's saw we have a simulation where we inject a current source at position `source` and measure the electric field intensity at `probe`.

Between these two points, there's a box at location `pos_box` with permittivity `eps`.

We're interested in computing how the intensity measured changes with respect to `eps`.

With ceviche, we first write a simple function computing the measured intensity as a function of `eps` using FDFD

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
    I = np.square(np.abs(Ex)) + np.square(np.abs(Ey))
    return = np.sum(I * probe)
```

Then, we can easily take the derivative of the intensity with respect to `eps` using a ceviche function


```python

# use autograd to differentiate `intensity` function
grad_fn = ceviche.jacobian(intensity)

# then, evaluate it at the current value of `eps`
dI_deps = grad_fn(eps_curr)

```

The beauty is that ceviche lets you compute this derivative without having to do any calculations by hand!  Using automatic differentiation, each step of the calculated is recorded and its derivative information is already known.  This lets us take derivatives of arbitrary complex code, where the output depends in some way on the electromagnetic simulation.

Armed with this capability, we can now do things like performing gradient-based optimization (inverse design) to maximize the intensity.

```python
for _ in range(10):
    eps_current += step_size * dI_deps_fn(eps_current)
```

It's also worth noting that the mathematics behind this gradient implementation uses the 'adjoint method', which lets you take derivatives with several degrees of freedom.  This is perfect for inverse design problems, or training of machine learning models that involve running an FDFD or FDTD simulation.  If you're interested in the connection between adjoint methods and backpropagation in the context of photonics, check out our group's earlier work on the subject [link](https://www.osapublishing.org/optica/abstract.cfm?uri=optica-5-7-864#articleMetrics).

## Installation

There are many ways to install `ceviche`.

The easiest is by 

    pip install ceviche

But to install from a local copy, one can instead do

    git clone https://github.com/twhughes/ceviche.git
    pip install -e ceviche
    pip install -r ceviche/requirements.txt

from the main directory.

Alternatively, just download it:

    git clone https://github.com/twhughes/ceviche.git

and then import the package from within your python script
    
```python
import sys
sys.path.append('path/to/ceviche')
```

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

    chmod + test/test_all_gradients.sh
    ./tests/test_all_gradients.sh

## Citation

If you use this for your research or work, please cite

    @misc{1908.10507,
    Author = {Tyler W Hughes and Ian A D Williamson and Momchil Minkov and Shanhui Fan},
    Title = {Forward-Mode Differentiation of Maxwell's Equations},
    Year = {2019},
    Eprint = {arXiv:1908.10507},
    }
