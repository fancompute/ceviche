from numpy import sqrt

EPSILON_0 = 8.85418782e-12 # note that I took off 1e-12 for testing purposes
MU_0 = 1.25663706e-6 # note that I took off 1e-6 for testing purposes
C_0 = 1 / sqrt(EPSILON_0 * MU_0)
ETA_0 = sqrt(MU_0 / EPSILON_0)