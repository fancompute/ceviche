# ceviche [![Build Status](https://travis-ci.com/fancompute/ceviche.svg?token=ZCPktA3Ki2eYVXYnfbrz&branch=master)](https://travis-ci.com/twhughes/ceviche)

Electromagnetic Simulation Tools + Automatic Differentiation.  Code for paper [Forward-Mode Differentiation of Maxwell's Equations](https://arxiv.org/abs/1908.10507).

<img src="/img/horizontal-color.png" title="ceviche" alt="ceviche">

## What is ceviche?

`ceviche` provides two core electromagnetic simulation tools for solving Maxwell's equations:

- finite-difference frequency-domain (FDFD)

- finite-difference time-domain (FDTD)

Both are written in `numpy` / `scipy` and are compatible with the [HIPS autograd package](https://github.com/HIPS/autograd), supporting forward-mode and reverse-mode automatic differentiation.

This allows you to write code to solve your E&M problem, and then use automatic differentiation on your results.

As a result, you can do gradient-based optimization, sensitivity analysis, or plug your E&M solver into a machine learning model without having to go through the tedious process of deriving your derivatives by hand.

## Examples

There is a comprehensive ceviche tutorial available at [this link](https://github.com/fancompute/workshop-invdesign) with several ipython notebook examples:
1. [Running FDFD simulations in ceviche.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/01_First_simulation.ipynb)
2. [Performing inverse design of a mode converter.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/02_Invdes_intro.ipynb)
3. [Adding fabrication constraints and device parameterizations.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/03_Invdes_parameterization.ipynb)
4. [Inverse design of a wavelength-division multiplexer and advanced topics.](https://nbviewer.jupyter.org/github/fancompute/workshop-invdesign/blob/master/04_Invdes_wdm_scheduling.ipynb)

There are also a few examples in the `examples/*` directory.

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

To get the FDFD and FDTD simulators, import directly `from ceviche import fdtd, fdfd_ez, fdfd_hz`

To get the differentiation, import `from ceviche import jacobian`.

`constants.py` contains some constants `EPSILON_0`, `C_0`, `ETA_0`, `Q_E`, which are needed throughout the package

`utils.py` contains a few useful functions for plotting, autogradding, and various other things.

`optimizers.py` contains optimizer functions for doing inverse design.

`viz.py` are functions that help with plotting fields and sructures.

`modes.py` contains a mode sorter (WIP) that can be used to create waveguide mode profiles for the simulation, for example.

### Examples

There are many demos in the `examples` directory, which will give you a good sense of how to use the package.

### Tests

Tests are located in `tests`.  To run, `cd` into `tests` and

    python -m unittest

to run all or

    python specific_test.py

to run a specific one.  Some of these tests involve visual inspection of the field plots rather than error checking on values.

To run all of the gradient checking functions, run 

    chmod +x test/test_all_gradients.sh
    tests/test_all_gradients.sh

## Credits

If you use this for your research or work, please cite

    @article{hughes2019forward,
      title={Forward-Mode Differentiation of Maxwell’s Equations},
      author={Hughes, Tyler W and Williamson, Ian AD and Minkov, Momchil and Fan, Shanhui},
      journal={ACS Photonics},
      volume={6},
      number={11},
      pages={3010--3016},
      year={2019},
      publisher={ACS Publications}
    }

Our logo was created by [@nagilmer](http://nadinegilmer.com/)
