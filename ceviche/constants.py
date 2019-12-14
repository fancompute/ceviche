from numpy import sqrt

"""
This file contains constants that are used throghout the codebase
"""

EPSILON_0 = 8.85418782e-12        # vacuum permittivity
MU_0 = 1.25663706e-6              # vacuum permeability
C_0 = 1 / sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum
ETA_0 = sqrt(MU_0 / EPSILON_0)    # vacuum impedance
Q_e = 1.602176634e-19             # funamental charge