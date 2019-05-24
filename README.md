# ceviche
Electromagnetic Simulation Tools + Automatic Differentiation

## User Guide


### FDFD

`fdfdpy.py` defines the FDFD stuff.  There is an `fdfd` base class and `fdfd_ez` and `fdfd_hz` subclasses.  This is organizationally a bit easier than using switching statements within a single FDFD like in `angler`.  Instead we just overload the basic `solve` and other operations for the different polarizations and handle the setup and other things in `fdfd`.



The meat is in `primitives.py`.