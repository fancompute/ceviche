import autograd.numpy as npa
import numpy as np
import numba as nb


def curl_E_numpy(axis, Ex, Ey, Ez, dLx, dLy, dLz):
    if axis == 0:
        return (npa.roll(Ez, shift=-1, axis=1) - Ez) / dLy - (npa.roll(Ey, shift=-1, axis=2) - Ey) / dLz
    elif axis == 1:
        return (npa.roll(Ex, shift=-1, axis=2) - Ex) / dLz - (npa.roll(Ez, shift=-1, axis=0) - Ez) / dLx
    elif axis == 2:
        return (npa.roll(Ey, shift=-1, axis=0) - Ey) / dLx - (npa.roll(Ex, shift=-1, axis=1) - Ex) / dLy

def curl_H_numpy(axis, Hx, Hy, Hz, dLx, dLy, dLz):
    if axis == 0:
        return (Hz - npa.roll(Hz, shift=1, axis=1)) / dLy - (Hy - npa.roll(Hy, shift=1, axis=2)) / dLz
    elif axis == 1:
        return (Hx - npa.roll(Hx, shift=1, axis=2)) / dLz - (Hz - npa.roll(Hz, shift=1, axis=0)) / dLx
    elif axis == 2:
        return (Hy - npa.roll(Hy, shift=1, axis=0)) / dLx - (Hx - npa.roll(Hx, shift=1, axis=1)) / dLy
