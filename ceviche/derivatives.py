import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from ceviche.constants import *

"""
This file contains functions related to performing derivative operations used in the simulation tools.
-  The FDTD method requires autograd-compatible curl operations, which are performed using numpy.roll
-  The FDFD method requires sparse derivative matrices, with PML added, which are constructed here.
"""


"""================================== CURLS FOR FDTD ======================================"""

def curl_E(axis, Ex, Ey, Ez, dL):
    if axis == 0:
        return (npa.roll(Ez, shift=-1, axis=1) - Ez) / dL - (npa.roll(Ey, shift=-1, axis=2) - Ey) / dL
    elif axis == 1:
        return (npa.roll(Ex, shift=-1, axis=2) - Ex) / dL - (npa.roll(Ez, shift=-1, axis=0) - Ez) / dL
    elif axis == 2:
        return (npa.roll(Ey, shift=-1, axis=0) - Ey) / dL - (npa.roll(Ex, shift=-1, axis=1) - Ex) / dL

def curl_H(axis, Hx, Hy, Hz, dL):
    if axis == 0:
        return (Hz - npa.roll(Hz, shift=1, axis=1)) / dL - (Hy - npa.roll(Hy, shift=1, axis=2)) / dL
    elif axis == 1:
        return (Hx - npa.roll(Hx, shift=1, axis=2)) / dL - (Hz - npa.roll(Hz, shift=1, axis=0)) / dL
    elif axis == 2:
        return (Hy - npa.roll(Hy, shift=1, axis=0)) / dL - (Hx - npa.roll(Hx, shift=1, axis=1)) / dL

"""======================= STUFF THAT CONSTRUCTS THE DERIVATIVE MATRIX ==========================="""

def compute_derivative_matrices(omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0):
    """ Returns sparse derivative matrices.  Currently works for 2D and 1D 
            omega: angular frequency (rad/sec)
            shape: shape of the FDFD grid
            npml: list of number of PML cells in x and y.
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
    """

    # Construct derivate matrices without PML
    Dxf = createDws('x', 'f', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dxb = createDws('x', 'b', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyf = createDws('y', 'f', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)
    Dyb = createDws('y', 'b', shape, dL, bloch_x=bloch_x, bloch_y=bloch_y)

    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb) = create_S_matrices(omega, shape, npml, dL)

    # apply PML to derivative matrices
    Dxf = Sxf.dot(Dxf)
    Dxb = Sxb.dot(Dxb)
    Dyf = Syf.dot(Dyf)
    Dyb = Syb.dot(Dyb)
    return Dxf, Dxb, Dyf, Dyb

def compute_derivative_matrices_3D(omega, shape, npml, dL, bloch_x=0.0, bloch_y=0.0, bloch_z=0.0):
    """ Returns sparse derivative matrices.  Currently works for 3D 
            omega: angular frequency (rad/sec)
            shape: shape of the FDFD grid
            npml: list of number of PML cells in x, y and z.
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
            block_z: bloch phase (phase across periodic boundary) in z
    """
    Nx, Ny , Nz= shape
    M = Nx*Ny*Nz
    Dex = make_Dex_3D(dL, shape, bloch_x)
    Dey = make_Dey_3D(dL, shape, bloch_y)
    print('Dey.shape',Dey.shape)
    Dez = make_Dez_3D(dL, shape, bloch_z)
    Dhx = make_Dhx_3D(dL, shape, bloch_x)
    Dhy = make_Dhy_3D(dL, shape, bloch_y)
    Dhz = make_Dhz_3D(dL, shape, bloch_z)
    print('Dhz.shape',Dhz.shape)
    Z = sp.csc_matrix((M, M))
    
    # make the S-matrices for PML
    (Sxf, Sxb, Syf, Syb, Szf, Szb) = create_S_matrices_3D(omega, shape, npml, dL)

    # apply PML to derivative matrices
    Dex = Sxf.dot(Dex)
    Dhx = Sxb.dot(Dhx)
    Dey = Syf.dot(Dey)
    Dhy = Syb.dot(Dhy)
    print('Dhy.shape',Dhy.shape)
    Dez = Szf.dot(Dez)
    print('Dez.shape',Dez.shape)
    Dhz = Szb.dot(Dhz)
    
    Ce1 = sp.hstack([Z, -Dez, Dey])
    Ce2 = sp.hstack([Dez, Z, -Dex])
    Ce3 = sp.hstack([-Dey,Dex,Z])
    Ce = sp.vstack([Ce1,Ce2,Ce3])
    Ch1 = sp.hstack([Z, -Dhz, Dhy])
    Ch2 = sp.hstack([Dhz, Z, -Dhx])
    Ch3 = sp.hstack([-Dhy,Dhx,Z])
    Ch = sp.vstack([Ch1,Ch2,Ch3])
    return Ce,Ch


""" Derivative Matrices (no PML) """

def createDws(component, dir, shape, dL, bloch_x=0.0, bloch_y=0.0):
    """ creates the derivative matrices
            component: one of 'x' or 'y' for derivative in x or y direction
            dir: one of 'f' or 'b', whether to take forward or backward finite difference
            shape: shape of the FDFD grid
            dL: spatial grid size (m)
            block_x: bloch phase (phase across periodic boundary) in x
            block_y: bloch phase (phase across periodic boundary) in y
    """

    Nx, Ny = shape    

    # special case, a 1D problem
    if component == 'x' and Nx == 1:
        return sp.eye(Ny)
    if component is 'y' and Ny == 1:
        return sp.eye(Nx)

    # select a `make_D` function based on the component and direction
    component_dir = component + dir
    if component_dir == 'xf':
        return make_Dxf(dL, shape, bloch_x=bloch_x)
    elif component_dir == 'xb':
        return make_Dxb(dL, shape, bloch_x=bloch_x)
    elif component_dir == 'yf':
        return make_Dyf(dL, shape, bloch_y=bloch_y)
    elif component_dir == 'yb':
        return make_Dyb(dL, shape, bloch_y=bloch_y)
    else:
        raise ValueError("component and direction {} and {} not recognized".format(component, dir))


def make_Dxf(dL, shape, bloch_x=0.0):
    """ Forward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxf = sp.diags([-1, 1, phasor_x], [0, 1, -Nx+1], shape=(Nx, Nx), dtype=np.complex128)
    Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny))
    return Dxf

def make_Dxb(dL, shape, bloch_x=0.0):
    """ Backward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxb = sp.diags([1, -1, -np.conj(phasor_x)], [0, -1, Nx-1], shape=(Nx, Nx), dtype=np.complex128)
    Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny))
    return Dxb

def make_Dyf(dL, shape, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyf = sp.diags([-1, 1, phasor_y], [0, 1, -Ny+1], shape=(Ny, Ny))
    Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf)
    return Dyf

def make_Dyb(dL, shape, bloch_y=0.0):
    """ Backward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyb = sp.diags([1, -1, -np.conj(phasor_y)], [0, -1, Ny-1], shape=(Ny, Ny))
    Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb)
    return Dyb


def make_Dex_3D(dL, shape, bloch_x=0.0):
    """ Forward derivative in x """
    Nx, Ny , Nz= shape
    phasor_x = np.exp(1j * bloch_x)
    Dex = sp.diags([-1, 1, phasor_x], [0, Nz*Ny, -Nx*Ny*Nz+Nz*Ny], shape=(Nx*Ny*Nz, Nx*Ny*Nz))
    Dex = 1 / dL * sp.kron(sp.eye(1),Dex)
    return Dex

def make_Dhx_3D(dL, shape, bloch_x=0.0):
    """ Backward derivative in x """
    Nx, Ny , Nz= shape
    phasor_x = np.exp(1j * bloch_x)
    Dhx = sp.diags([1, -1, -phasor_x], [0, -Nz*Ny, Nx*Ny*Nz-Nz*Ny], shape=(Nx*Ny*Nz, Nx*Ny*Nz))
    Dhx = 1 / dL * sp.kron(sp.eye(1),Dhx)
    return Dhx
    
def make_Dey_3D(dL, shape, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny , Nz= shape
    phasor_y = np.exp(1j * bloch_y)
    Dey = sp.diags([-1, 1, phasor_y], [0, Nz, -Nz*Ny+Nz], shape=(Nz*Ny, Nz*Ny))
    Dey = 1 / dL * sp.kron(sp.eye(Nx),Dey)
    return Dey

def make_Dhy_3D(dL, shape, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny , Nz= shape
    phasor_y = np.exp(1j * bloch_y)
    Dhy = sp.diags([1, -1, -phasor_y], [0, -Nz, Nz*Ny-Nz], shape=(Nz*Ny, Nz*Ny))
    Dhy = 1 / dL * sp.kron(sp.eye(Nx),Dhy)
    return Dhy
    
def make_Dez_3D(dL, shape, bloch_z=0.0):
    """ Forward derivative in z """
    Nx, Ny , Nz= shape
    phasor_z = np.exp(1j * bloch_z)
    Dez = sp.diags([-1, 1, phasor_z], [0, 1, -Nz+1], shape=(Nz, Nz))
    Dez = 1 / dL * sp.kron(sp.eye(Nx*Ny),Dez)
    return Dez

def make_Dhz_3D(dL, shape, bloch_z=0.0):
    """ Forward derivative in z """
    Nx, Ny , Nz= shape
    phasor_z = np.exp(1j * bloch_z)
    Dhz = sp.diags([1, -1, -phasor_z], [0, -1, Nz-1], shape=(Nz, Nz))
    Dhz = 1 / dL * sp.kron(sp.eye(Nx*Ny),Dhz)
    return Dhz
    
""" PML Functions """

def create_S_matrices(omega, shape, npml, dL):
    """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    x_range = [0, float(dL * Nx)]
    y_range = [0, float(dL * Ny)]
    Nx_pml, Ny_pml = npml    

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(shape, dtype=np.complex128)
    Sx_b_2D = np.zeros(shape, dtype=np.complex128)
    Sy_f_2D = np.zeros(shape, dtype=np.complex128)
    Sy_b_2D = np.zeros(shape, dtype=np.complex128)

    # insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1 / s_vector_x_f
        Sx_b_2D[:, i] = 1 / s_vector_x_b
    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1 / s_vector_y_f
        Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.flatten()
    Sx_b_vec = Sx_b_2D.flatten()
    Sy_f_vec = Sy_f_2D.flatten()
    Sy_b_vec = Sy_b_2D.flatten()

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)

    return Sx_f, Sx_b, Sy_f, Sy_b

def create_sfactor(dir, omega, dL, N, N_pml):
    """ creates the S-factor cross section needed in the S-matrices """

    #  for no PNL, this should just be zero
    if N_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # otherwise, get different profiles for forward and reverse derivative matrices
    dw = N_pml * dL
    if dir == 'f':
        return create_sfactor_f(omega, dL, N, N_pml, dw)
    elif dir == 'b':
        return create_sfactor_b(omega, dL, N, N_pml, dw)
    else:
        raise ValueError("Dir value {} not recognized".format(dir))

def create_sfactor_f(omega, dL, N, N_pml, dw):
    """ S-factor profile for forward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
    return sfactor_array
    
def create_sfactor_b(omega, dL, N, N_pml, dw):
    """ S-factor profile for backward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_array

def sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

def s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    return 1 - 1j * sig_w(l, dw) / (omega * EPSILON_0)

def create_S_matrices_3D(omega, shape, npml, dL):
    """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

    # strip out some information needed
    Nx, Ny, Nz = shape
    #print(shape, Nx, Ny, Nz)
    N = Nx * Ny * Nz
    x_range = [0, float(dL * Nx)]
    y_range = [0, float(dL * Ny)]
    z_range = [0, float(dL * Nz)]
    Nx_pml = npml[0]
    Ny_pml = npml[1]
    Nz_pml = npml[2]
    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)
    s_vector_z_f = create_sfactor('f', omega, dL, Nz, Nz_pml)
    s_vector_z_b = create_sfactor('b', omega, dL, Nz, Nz_pml)
    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_3D = np.zeros(shape, dtype=np.complex128)
    Sx_b_3D = np.zeros(shape, dtype=np.complex128)
    Sy_f_3D = np.zeros(shape, dtype=np.complex128)
    Sy_b_3D = np.zeros(shape, dtype=np.complex128)
    Sz_f_3D = np.zeros(shape, dtype=np.complex128)
    Sz_b_3D = np.zeros(shape, dtype=np.complex128)
    # insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(0,Ny):
        for k in range(0,Nz):
            Sx_f_3D[:,i,k] = 1/s_vector_x_f
            Sx_b_3D[:,i,k] = 1/s_vector_x_b
    for i in range(0,Nx):
        for k in range(0, Nz):
            Sy_f_3D[i,:,k] = 1/s_vector_y_f
            Sy_b_3D[i,:,k] = 1/s_vector_y_b
    for i in range(0, Nx):
        for k in range(0, Ny):
            Sz_f_3D[i,k,:] = 1/s_vector_z_f
            Sz_b_3D[i,k,:] = 1/s_vector_z_b
    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_3D.flatten()
    Sx_b_vec = Sx_b_3D.flatten()
    Sy_f_vec = Sy_f_3D.flatten()
    Sy_b_vec = Sy_b_3D.flatten()
    Sz_f_vec = Sy_f_3D.flatten()
    Sz_b_vec = Sy_b_3D.flatten()
    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N)
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N)
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N)
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N)
    Sz_f = sp.spdiags(Sz_f_vec, 0, N, N)
    Sz_b = sp.spdiags(Sz_b_vec, 0, N, N)
    return Sx_f, Sx_b, Sy_f, Sy_b, Sz_f, Sz_b
