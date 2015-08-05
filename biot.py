from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, mgrid
from mayavi import mlab
NGRID=25
x=y=z=np.linspace(-5,5,NGRID)

grid_positions=np.zeros((NGRID**3,3))
for ix, vx in enumerate(x):
    for iy, vy in enumerate(y):
        for iz, vz in enumerate(z):
            row = NGRID**2*ix+NGRID*iy+iz
            grid_positions[row, 0] = vx
            grid_positions[row, 1] = vy
            grid_positions[row, 2] = vz

grid_B=np.zeros_like(grid_positions)

N_wires=6
r_wires = 2.

for i in range(N_wires):
	print("wire " +str(i))
    angle = 2*i*np.pi/N_wires
    x_wire_pos=r_wires*np.cos(angle)
    y_wire_pos=r_wires*np.sin(angle)
    N=100
    z_wire=np.linspace(-5,5,N)
    x_wire=np.ones_like(z_wire)*x_wire_pos
    y_wire=np.ones_like(z_wire)*y_wire_pos
    
    wire_current = 1
    wire = np.vstack((x_wire, y_wire, z_wire)).T
    wire_gradient = np.gradient(wire)[0]
    wire_length = np.sqrt(np.sum(wire_gradient**2, axis=1))
    for index, wire_segment in enumerate(wire):
        wire_segment_length = wire_gradient[index,:]*wire_length[index]
        rprime=(grid_positions-wire_segment)
        distances = np.sum(rprime**2, axis=1)**(3./2.)
        denominator = np.vstack((distances, distances, distances)).T
        differential=np.cross(wire_segment_length, rprime)/denominator*wire_current*1e7

        low_cutoff_indices=distances<0.01
        indices_cut_off=np.sum(low_cutoff_indices)
        if(indices_cut_off>0):
            differential[low_cutoff_indices, :] = 0
        grid_B += differential
    grid_B[np.isinf(grid_B)] = np.nan
    mlab.plot3d(x_wire,y_wire,z_wire)


display_every_n_point=1
x_display=grid_positions[::display_every_n_point,0]
y_display=grid_positions[::display_every_n_point,1]
z_display=grid_positions[::display_every_n_point,2]
bx_display=grid_B[::display_every_n_point,0]
by_display=grid_B[::display_every_n_point,1]
bz_display=grid_B[::display_every_n_point,2]
B_magnitude_squared=np.sqrt(np.sum(grid_B**2, axis=1))

mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display)
#mlab.points3d(x_display,y_display,z_display, B_magnitude_squared[::display_every_n_point])
mlab.show()
