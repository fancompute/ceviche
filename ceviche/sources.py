import autograd.numpy as npa

""" Source functions go here.  For now it's just TFSF """

def b_TFSF(fdfd, inside_mask, theta):
    """ Returns a source vector for FDFD that will implement TFSF 
            A: the FDFD system matrix
            inside_mask: a binary mask (vector) specifying the inside of the TFSF region
            theta: [0, 2pi] the angle of the source relative to y=0+

                      y ^
                        |
                        |
                  <-----|----- > x
                        |\
                        | \                                     
                        v |\
                          theta             
    see slide 32 of https://empossible.net/wp-content/uploads/2019/08/Lecture-4d-FDFD-Formulation.pdf                                                       
    """

    lambda0 = 2 * npa.pi * C_0 / fdfd.omega
    f_src = compute_f(theta, lambda0, fdfd.dL,  inside.shape)

    Q = compute_Q(inside_mask) / fdfd.omega # why this omega??
    A = fdfd.make_A(fdfd.eps_r.copy().flatten())

    quack = (Q.dot(A) - A.dot(Q))

    return quack.dot(f_src)

def compute_Q(inside_mask):
    """ Compute the matrix used in PDF to get source """

    # convert masks to vectors and get outside portion
    inside_vec = inside_mask.flatten()
    outside_vec = 1 - inside_vec
    N = outside_vec.size

    # make a sparse diagonal matrix and return
    Q = sp.diags([outside_vec], [0], shape=(N, N))
    return Q


def compute_f(theta, lambda0, dL, shape):
    """ Compute the 'vacuum' field vector """

    # get plane wave k vector components (in units of grid cells)
    k0 = 2 * npa.pi / lambda0 * dL
    kx =  k0 * npa.sin(theta)
    ky = -k0 * npa.cos(theta)  # negative because downwards

    # array to write into
    f_src = npa.zeros(shape, dtype=npa.complex128)

    # get coordinates
    Nx, Ny = shape
    xpoints = npa.arange(Nx)
    ypoints = npa.arange(Ny)
    xv, yv = npa.meshgrid(xpoints, ypoints, indexing='ij')

    # compute values and insert into array
    x_PW = npa.exp(1j * xpoints * kx)[:, None]
    y_PW = npa.exp(1j * ypoints * ky)[:, None]

    f_src[xv, yv] = npa.outer(x_PW, y_PW)

    return f_src.flatten()
