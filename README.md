# ceviche
Electromagnetic Simulation Tools + Automatic Differentiation

## What is Ceviche?

`ceviche` provides two core electromagnetic simulation tools for solving Maxwell's equations:

- finite-difference frequency-domain (FDFD)

- finite-difference time-domain (FDTD)

Both are written in `numpy` / `scipy` and are compatible with the [HIPS autograd package](https://github.com/HIPS/autograd).

What this means is that you can write code to solve your E&M problem, and then use automatic differentiation on your results.

This is incredibly powerful as it allows you to do gradient-based optimization or sensitivity analysis without the tedius process of deriving your derivatives analytically.

### A quick example

Lets say we have a domain of where we wish to inject light at position `source` and measure it's intensity at `measure`.

Between these two points, there's a box at location `pos_box` with permittivity `eps`.

We can write a function computing the intensity as a function of `eps` using our FDFD solver

```python
def intensity(eps):
    """ computes electric intensity at `probe` for a given box permittivity of `eps`

        source |-----| probe
            .  | eps |  .
               |_____|

    `fdfd` is a ceviche FDFD object that is defined earlier for this problem ^
    """

    # set the permittivity in the box region to the input argument
    fdfd.eps_r[box_pos] = eps

    # solve the fields
    Ex, Ey, Hz = fdfd.solve(source)

    # compute the intensity at `probe`
    I = np.square(npa.abs(Ex)) + npa.square(npa.abs(Ex))
    return = np.sum(I * probe)
```

Then, we can very easily differentiate this function using automatic differentiation

```python

# use autograd to differentiate `intensity` function
dI_deps_fn = autograd.grad(intensity)

# evaluate it at the current value of `eps`
dI_deps = dI_deps_fn(eps_current)

# or ... do gradient based optimization
for _ in range(10):
    eps_current += step_size * dI_deps_fn(eps_current)
```

