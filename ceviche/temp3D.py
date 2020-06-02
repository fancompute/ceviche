import numpy as np
from mayavi import mlab
import autograd.numpy as npa
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
import ceviche
from ceviche.constants import C_0
# Source angular frequency in Hz
f = 200e12
#omega = 2*np.pi*f
omega = 2*np.pi*200e12
lamda = C_0/f
print(lamda)
# Resolution in nm
dl = 20e-9
# Simulation domain size in number of pixels
Nx = 80
Ny = 80
Nz = 3
# Size of the PML boundaries in pixels 
Npml = [20, 20, 1]
# Initialize relative permittivity of the domain
epsr = np.ones((Nx, Ny, Nz))  
# Set the permittivity to 12 inside a box
#epsr[12,12,5] = 12
#Initialize the source position and amplitude
src_x = np.arange(40,60)
src_y = 40 * np.ones(src_x.shape, dtype=int)
src_z = 2 * np.ones(src_x.shape, dtype=int)
sourcex = np.zeros((Nx, Ny, Nz))
sourcey = np.zeros((Nx, Ny, Nz))
sourcez = np.zeros((Nx, Ny, Nz))
#sourcex[src_x, src_y,src_z] = 1000
#sourcey[src_x, src_y,src_z] = 1000
sourcez[src_x, src_y,src_z] = 1000
source = npa.hstack((sourcex.flatten(),sourcey.flatten(),sourcez.flatten()))
print(npa.max(source))
# Create the simulation object for 'Ez' (TE) polarization
simulation = ceviche.fdfd3D(omega, dl, epsr, Npml)
# Run the simulation with the given source
E = simulation.solve(source)
Ex, Ey, Ez = simulation.Vec3Dtogrid(E)

print(np.max(abs(E)))
print(np.max(abs(Ex)))
#print(np.max(abs(Ey)))
#print(np.max(abs(Ez)))
#print(Ez.shape)
#E = np.sqrt(Ex*Ex+Ey*Ey+Ez*Ez)
s = npa.abs(Ez)
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='x_axes',
                            slice_index=20)
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                            plane_orientation='y_axes',
                            slice_index=20)
mlab.pipeline.image_plane_widget(mlab.pipeline.scalar_field(s),
                           plane_orientation='z_axes',
                           slice_index=20)
mlab.outline()
mlab.show()

print('END')