import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d

NGRID=25
x=y=z=np.linspace(-1,1,NGRID)

grid_positions=np.zeros((NGRID**3,3))
for ix, vx in enumerate(x):
    for iy, vy in enumerate(y):
        for iz, vz in enumerate(z):
            row = NGRID**2*ix+NGRID*iy+iz
            grid_positions[row, 0] = vx
            grid_positions[row, 1] = vy
            grid_positions[row, 2] = vz

theta=np.linspace(0,2*np.pi, 1000)
x_wire = 0.5*np.cos(theta)
y_wire = 0.5*np.sin(theta)
z_wire = np.zeros_like(x_wire)

# N=1000
# z_wire=np.linspace(-1,1,N)
# x_wire=y_wire=np.zeros_like(z_wire)


wire_current = 1
wire = np.vstack((x_wire, y_wire, z_wire)).T
wire_gradient = np.gradient(wire)[0]
wire_length = np.sqrt(np.sum(wire_gradient**2, axis=1))

grid_B=np.zeros_like(grid_positions)
for index, wire_segment in enumerate(wire):
    wire_segment_length = wire_gradient[index,:]
    rprime=(grid_positions-wire_segment)
    differential=np.cross(wire_segment_length, rprime)/np.abs(rprime)**3
    grid_B += differential
grid_B*=wire_current*1e7
print(grid_B)

# for i in range(NGRID):
#     x_display = grid_positions[i::NGRID,0]
#     y_display = grid_positions[i::NGRID,1]
#     bx_display = grid_B[i::NGRID,0]
#     by_display = grid_B[i::NGRID,1]
#     plt.quiver(x_display,y_display,bx_display, by_display)
#     plt.plot()
#     plt.show()

coktory=5
x_display=grid_positions[::coktory,0]
y_display=grid_positions[::coktory,1]
z_display=grid_positions[::coktory,2]
bx_display=grid_B[::coktory,0]
by_display=grid_B[::coktory,1]
bz_display=grid_B[::coktory,2]
fig = plt.figure()
ax=fig.gca(projection='3d')
ax.plot(x_wire,y_wire,z_wire, "r-")
ax.quiver(x_display, y_display, z_display, bx_display, by_display, bz_display,
    length=0.1)
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-1,1)
plt.show()
