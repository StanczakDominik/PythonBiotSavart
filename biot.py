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

theta=np.linspace(0,2*np.pi, 1000)
x_wire = 2.5*np.cos(theta*4)
y_wire = 2.5*np.sin(theta*4)
# z_wire = np.zeros_like(x_wire)
z_wire = np.linspace(-4.9,4.9,1000)

# N=100
# z_wire=np.linspace(-5,5,N)
# x_wire=y_wire=np.zeros_like(z_wire)


wire_current = 1
wire = np.vstack((x_wire, y_wire, z_wire)).T
wire_gradient = np.gradient(wire)[0]
wire_length = np.sqrt(np.sum(wire_gradient**2, axis=1))

grid_B=np.zeros_like(grid_positions)
for index, wire_segment in enumerate(wire):
    wire_segment_length = wire_gradient[index,:]*wire_length[index]
    rprime=(grid_positions-wire_segment)
    distances = np.sum(rprime**2, axis=1)**(3/2)
    denominator = np.vstack((distances, distances, distances)).T
    differential=np.cross(wire_segment_length, rprime)/denominator
    grid_B += differential
grid_B*=wire_current*1e7
grid_B[np.isinf(grid_B)]=np.nan
print(grid_B)

coktory=5
x_display=grid_positions[::coktory,0]
y_display=grid_positions[::coktory,1]
z_display=grid_positions[::coktory,2]
bx_display=grid_B[::coktory,0]
by_display=grid_B[::coktory,1]
bz_display=grid_B[::coktory,2]

mlab.plot3d(x_wire,y_wire,z_wire)
B_magnitude_squared=np.sqrt(np.sum(grid_B**2, axis=1))
mlab.quiver3d(x_display, y_display, z_display, bx_display, by_display, bz_display)
#mlab.points3d(x_display,y_display,z_display, B_magnitude_squared[::coktory])
mlab.show()
