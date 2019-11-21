# ceviche [![Build Status](https://travis-ci.com/fancompute/ceviche.svg?token=ZCPktA3Ki2eYVXYnfbrz&branch=master)](https://travis-ci.com/fancompute/ceviche)

Electromagnetic Simulation Tools + Automatic Differentiation.  Code for the paper: [Forward-Mode Differentiation of Maxwell's Equations](https://pubs.acs.org/doi/abs/10.1021/acsphotonics.9b01238) ([arxiv preprint](https://arxiv.org/abs/1908.10507)).

<img src="/img/horizontal-color.png" title="ceviche" alt="ceviche">

(logo by [@ngilmer](http://nadinegilmer.com/))

## What is ceviche?

`ceviche` provides two core electromagnetic simulation tools for solving Maxwell's equations:

- finite-difference frequency-domain (FDFD)

- finite-difference time-domain (FDTD)

Both are written in `numpy` / `scipy` and are compatible with the [HIPS autograd package](https://github.com/HIPS/autograd), supporting forward-mode and reverse-mode automatic differentiation.

This allows you to write code to solve your E&M problem, and then use automatic differentiation on your results.

As a result, you can do gradient-based optimization, sensitivity analysis, or plug your E&M solver into a machine learning model without the tedius process of deriving your derivatives analytically.

## Installation

There are many ways to install `ceviche`.

The easiest is by 

    pip install ceviche

But to install from a local copy, one can instead do

    git clone https://github.com/twhughes/ceviche.git
    pip install -e ceviche
    pip install -r ceviche/requirements.txt

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

There are many demos in the `examples` directory, which will give you a good sense of how to use the package. There is also workshop material consiting of a series of notebooks available in a [separate repository](https://github.com/fancompute/workshop-invdesign).

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

    @article{hughes2019forward,
      title={Forward-mode Differentiation of Maxwell's Equations},
      author={Hughes, Tyler W and Williamson, Ian AD and Minkov, Momchil and Fan, Shanhui},
      journal={ACS Photonics},
      year={2019},
      publisher={ACS Publications}
    }

