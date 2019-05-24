# ceviche
Electromagnetic Simulation Tools + Automatic Differentiation

## User Guide


### FDFD

`fdfdpy.py` defines the FDFD stuff.  There is an `fdfd` base class and `fdfd_ez` and `fdfd_hz` subclasses.  This is organizationally a bit easier than using switching statements within a single FDFD like in `angler`.  Instead we just overload the basic `solve` and other operations for the different polarizations and handle the setup and other things in `fdfd`.

This file also contains some functions that make the derivative operators and the PML.
This is where I'm confused.  I think the current code is kind of ugly and not very intuitive.
Perhaps we can move this to its own file, or `utils.py` and simplify the interface.

### Adjoint Stuff

The autograd meat is in `primitives.py`.  Here I define the A matrices in terms of the derivative operators and the permittivity.
I also define the conversion from Hz to Ex, Ey and same for Ez polarization.  I also define all of the primitives for taking derivatives with respect to epsilon_r.  However, we should add the primitive for the source as well, which should be simple

    E = A^-1 b  
    dE/db*v = A^{-1} v (or something, maybe transposed A)

The primitives are complicated and it took a long time to get them working.
When I try to add the constants in from `constants.py`, things don't match.
Also, when I try to use realistic units in the FDFD tests, I get really small field magnitudes and other weird behavior.
Note that I do not define an L0 because its messy in my opinion.  Everything is SI units.  However, we might need to define some normalization of A to make things nicer.

### Tests

`tests/test_grads.py` does the gradient checking on a simple problem.

`test/test_fields.py` plots some fields.

### Examples

`examples/DLA.py` does a simple DLA optimization example, and seems to work..

Let me know if there are any questions?

