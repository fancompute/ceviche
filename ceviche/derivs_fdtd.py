import autograd.numpy as npa
import numpy as np
import numba as nb

"""

This file contains various implementations of the curl operators in maxwell's equations.
They are imported and used in the forward() method of the FDTD object.
curl_{E,H}_numpy is the most stable and is also compatible with autograd.
ceviche defaults to using this curl.

Running this file as __main__ will run some interesting speed tests and make sure the implementations match each other.

"""

# These derivatives use np.roll to implement the curl,  They are optimized without using JIT.

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

# These functions operate on a vector E where the 5th dimension indeces into field component {x,y,z}.  
# This is just a slighly more compact version that I thought might have a speedup, but I didn't really observe much of a help from doing this.

def curl_E_vec(E, dLx, dLy, dLz):
    CEx = (npa.roll(E[:,:,:,:,2], shift=-1, axis=1) - E[:,:,:,:,2]) / dLy - (npa.roll(E[:,:,:,:,1], shift=-1, axis=2) - E[:,:,:,:,1]) / dLz
    CEy = (npa.roll(E[:,:,:,:,0], shift=-1, axis=2) - E[:,:,:,:,0]) / dLz - (npa.roll(E[:,:,:,:,2], shift=-1, axis=0) - E[:,:,:,:,2]) / dLx
    CEz = (npa.roll(E[:,:,:,:,1], shift=-1, axis=0) - E[:,:,:,:,1]) / dLx - (npa.roll(E[:,:,:,:,0], shift=-1, axis=1) - E[:,:,:,:,0]) / dLy
    return npa.concatenate((CEx[:,:,:,:,None], CEy[:,:,:,:,None], CEz[:,:,:,:,None]), axis=4)

def curl_H_vec(H, dLx, dLy, dLz):
    CHx = (H[:,:,:,:,2] - npa.roll(H[:,:,:,:,2], shift=1, axis=1)) / dLy - (H[:,:,:,:,1] - npa.roll(H[:,:,:,:,1], shift=1, axis=2)) / dLz
    CHy = (H[:,:,:,:,0] - npa.roll(H[:,:,:,:,0], shift=1, axis=2)) / dLz - (H[:,:,:,:,2] - npa.roll(H[:,:,:,:,2], shift=1, axis=0)) / dLx
    CHz = (H[:,:,:,:,1] - npa.roll(H[:,:,:,:,1], shift=1, axis=0)) / dLx - (H[:,:,:,:,0] - npa.roll(H[:,:,:,:,0], shift=1, axis=1)) / dLy
    return npa.concatenate((CHx[:,:,:,:,None], CHy[:,:,:,:,None], CHz[:,:,:,:,None]), axis=4)

# These functions use an explicit for loop
# They are slow AF, but when they are jit compiled with numba, they are much faster than above, especially for large batch sizes.
# howver, note that when called inside of the fdtd.run loop, overhead from switching between numba and python makes them about as fast as numpy curls in the end.

def curl_E_loop(axis, Ex, Ey, Ez, dLx, dLy, dLz):
    shape_in = (Nx, Ny, Nz, Nb) = Ex.shape
    E_out = np.zeros(shape_in)
    if axis == 0:
        for i in range(Nx):
            for j in range(Ny):
                j_plus = (j+1) % Ny
                for k in range(Nz):
                    k_plus = (k+1) % (Nz)
                    for l in range(Nb):
                        # dEz_dy - dEy_dz
                        E_out[i, j, k, l] = (Ez[i, j_plus, k, l] - Ez[i, j, k, l]) / dLy - (Ey[i, j, k_plus, l] - Ey[i, j, k, l]) / dLz
        return E_out

    if axis == 1:
        for i in range(Nx):
            i_plus = (i+1) % Nx
            for j in range(Ny):
                for k in range(Nz):
                    k_plus = (k+1) % Nz
                    for l in range(Nb):
                        # dEx_dz - dEz_dx
                        E_out[i, j, k, l] = (Ex[i, j, k_plus, l] - Ex[i, j, k, l]) / dLz - (Ez[i_plus, j, k, l] - Ez[i, j, k, l]) / dLx
        return E_out

    if axis == 2:
        for i in range(Nx):
            i_plus = (i+1) % Nx            
            for j in range(Ny):
                j_plus = (j+1) % Ny                
                for k in range(Nz):
                    for l in range(Nb):
                        # dEy_dx - dEx_dy
                        E_out[i, j, k, l] = (Ey[i_plus, j, k, l] - Ey[i, j, k, l]) / dLx - (Ex[i, j_plus, k, l] - Ex[i, j, k, l]) / dLy   
        return E_out

def curl_H_loop(axis, Hx, Hy, Hz, dLx, dLy, dLz):
    shape_in = (Nx, Ny, Nz, Nb) = Hx.shape
    H_out = np.zeros(shape_in)
    if axis == 0:
        for i in range(Nx):
            for j in range(Ny):
                j_minus = (j-1) % Ny
                for k in range(Nz):
                    k_minus = (k-1) % Nz
                    for l in range(Nb):
                        H_out[i, j, k, l] = (Hz[i, j, k, l] - Hz[i, j_minus, k, l]) / dLy - (Hy[i, j, k, l] - Hy[i, j, k_minus, l]) / dLz
        return H_out

    if axis == 1:
        for i in range(Nx):
            i_minus = (i-1) % Nx
            for j in range(Ny):
                for k in range(Nz):
                    k_minus = (k-1) % Nz
                    for l in range(Nb):
                        # dHx_dz - dHz_dx
                        H_out[i, j, k, l] = (Hx[i, j, k, l] - Hx[i, j, k_minus, l]) / dLz - (Hz[i, j, k, l] - Hz[i_minus, j, k, l]) / dLx
        return H_out

    if axis == 2:
        for i in range(Nx):
            i_minus = (i-1) % Nx
            for j in range(Ny):
                j_minus = (j-1) % Ny                
                for k in range(Nz):
                    for l in range(Nb):
                        # dHy_dx - dHx_dy
                        H_out[i, j, k, l] = (Hy[i, j, k, l] - Hy[i_minus, j, k, l]) / dLx - (Hx[i, j, k, l] - Hx[i, j_minus, k, l]) / dLy
        return H_out

# You can jit compile the above for loops with numba.

curl_E_numba = nb.njit(curl_E_loop)
curl_H_numba = nb.njit(curl_H_loop)

# These are the original ceviche curls, which are a little clunky and not as fast as np.roll

def periodic_deriv_E(E, axis):
    der = npa.zeros(E.shape)
    if axis == 'x':
        der = npa.roll(E, shift=-1, axis=0) - E
    elif axis == 'y':
        der = npa.roll(E, shift=-1, axis=1) - E
    elif axis == 'z':
        der = npa.roll(E, shift=-1, axis=2) - E
    return der

def periodic_deriv_H(H, axis):
    der = npa.zeros(H.shape)
    if axis == 'x':
        der = H - npa.roll(H, shift=1, axis=0)
    elif axis == 'y':
        der = H - npa.roll(H, shift=1, axis=1)
    elif axis == 'z':
        der = H - npa.roll(H, shift=1, axis=2)
    return der

def curl_E_old(axis, Ex, Ey, Ez, dLx, dLy, dLz):
    C = npa.zeros(Ex.shape)
    if axis == 0:
        # dEz_dy - dEy_dz
        C = C + periodic_deriv_E(Ez, 'y') / dLy
        C = C - periodic_deriv_E(Ey, 'z') / dLz
    if axis == 1:
        # dEx_dz - dEz_dx
        C = C + periodic_deriv_E(Ex, 'z') / dLz
        C = C - periodic_deriv_E(Ez, 'x') / dLx
    if axis == 2:
        # dEy_dx - dEx_dy
        C = C + periodic_deriv_E(Ey, 'x') / dLx
        C = C - periodic_deriv_E(Ex, 'y') / dLy
    return C

def curl_H_old(axis, Hx, Hy, Hz, dLx, dLy, dLz):
    C = npa.zeros(Hx.shape)
    if axis == 0:
        # dEz_dy - dEy_dz
        C = C + periodic_deriv_H(Hz, 'y') / dLy
        C = C - periodic_deriv_H(Hy, 'z') / dLz
    if axis == 1:
        # dEx_dz - dEz_dx
        C = C + periodic_deriv_H(Hx, 'z') / dLz
        C = C - periodic_deriv_H(Hz, 'x') / dLx
    if axis == 2:
        # dEy_dx - dEx_dy
        C = C + periodic_deriv_H(Hy, 'x') / dLx
        C = C - periodic_deriv_H(Hx, 'y') / dLy
    return C


if __name__ == '__main__':    

    """ TESTS that the numba and numpy curls are equal """

    print('setting up\n')

    shape = Nx, Ny, Nz, Nbatch = 100,100,10,10
    Ex = np.random.random(shape)
    Ey = np.random.random(shape)
    Ez = np.random.random(shape)
    Hx = np.random.random(shape)
    Hy = np.random.random(shape)
    Hz = np.random.random(shape)    
    dLx, dLy, dLz = 1,2,3
    import matplotlib.pylab as plt
    from time import time

    print('testing single curls with shape {}'.format(shape))

    # old
    t = time()
    A = curl_E_old(0, Ex, Ey, Ez, dLx, dLy, dLz)
    A = curl_H_old(0, Ex, Ey, Ez, dLx, dLy, dLz)
    print('    -> "old" took {} seconds'.format(time() - t))

    # numpy
    t = time()
    A = curl_E_numpy(0, Ex, Ey, Ez, dLx, dLy, dLz)
    A = curl_H_numpy(0, Ex, Ey, Ez, dLx, dLy, dLz)
    print('    -> "numpy" took {} seconds'.format(time() - t))

    # compile numba
    B = curl_E_numba(0, Ex, Ey, Ez, dLx, dLy, dLz)
    B = curl_H_numba(0, Ex, Ey, Ez, dLx, dLy, dLz)

    # numba
    t = time()
    B = curl_E_numba(0, Ex, Ey, Ez, dLx, dLy, dLz)
    B = curl_H_numba(0, Ex, Ey, Ez, dLx, dLy, dLz)
    print('    -> "numba" took {} seconds\n'.format(time() - t))

    print('testing that numpy = numba')
    np.testing.assert_allclose(curl_E_numpy(0, Ex, Ey, Ez, dLx, dLy, dLz), curl_E_numba(0, Ex, Ey, Ez, dLx, dLy, dLz))
    print('    -> Ex looks OK')
    np.testing.assert_allclose(curl_E_numpy(1, Ex, Ey, Ez, dLx, dLy, dLz), curl_E_numba(1, Ex, Ey, Ez, dLx, dLy, dLz))
    print('    -> Ey looks OK')
    np.testing.assert_allclose(curl_E_numpy(2, Ex, Ey, Ez, dLx, dLy, dLz), curl_E_numba(2, Ex, Ey, Ez, dLx, dLy, dLz))
    print('    -> Ez looks OK')

    np.testing.assert_allclose(curl_H_numpy(0, Hx, Hy, Hz, dLx, dLy, dLz), curl_H_numba(0, Hx, Hy, Hz, dLx, dLy, dLz))
    print('    -> Hx looks OK')
    np.testing.assert_allclose(curl_H_numpy(1, Hx, Hy, Hz, dLx, dLy, dLz), curl_H_numba(1, Hx, Hy, Hz, dLx, dLy, dLz))
    print('    -> Hy looks OK')
    np.testing.assert_allclose(curl_H_numpy(2, Hx, Hy, Hz, dLx, dLy, dLz), curl_H_numba(2, Hx, Hy, Hz, dLx, dLy, dLz))
    print('    -> Hz looks OK')

    # time simulations with different derivatives
    print('testing FDTD simulations with different curl implementations')
    from ceviche import FDTD
    from sources import Gaussian
    from plot import animate_field

    print('setting up with shape {}'.format(shape))

    steps = 10
    eps_r = np.ones(shape)
    eps_r[30:70, 30:70, :, :] = 5
    source_loc1 = np.zeros(eps_r.shape)
    source_loc1[40:60, 50, 0, :] = 1
    G1 = Gaussian(mask=source_loc1, component='Jz', amp=10, sigma=20, t0=100)

    print('starting simulation with "old" curl')

    F = FDTD(eps_r, dL=1e-8, NPML=[20, 20, 0], deriv='old')
    F.add_src(G1)
    t = time()
    for t_index, fields in enumerate(F.run(steps=steps)):
        A = fields
    print('    -> with "old" curl took {} seconds'.format(time() - t))

    print('starting simulation with "numpy" curl')
    F = FDTD(eps_r, dL=1e-8, NPML=[20, 20, 0], deriv='numpy')
    F.add_src(G1)
    t = time()
    for t_index, fields in enumerate(F.run(steps=steps)):
        A = fields
    print('    -> with "numpy" curl took {} seconds'.format(time() - t))

    print('starting simulation with "numba" curl')
    F = FDTD(eps_r, dL=1e-8, NPML=[20, 20, 0], deriv='numba')
    F.add_src(G1)
    t = time()
    for t_index, fields in enumerate(F.run(steps=steps)):
        A = fields
    print('    -> with "numba" curl took {} seconds'.format(time() - t))
