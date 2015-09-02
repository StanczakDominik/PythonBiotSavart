from biot import *

#Simulation parameters
N_iterations=int(1e6)
Dump_every_N_iterations=int(1e6)
N_particles=1
N_interpolation=8
velocity_scaling=1e6 #for random selection of initial velocity
dt=0.01/velocity_scaling

grid_positions, dx, dy, dz=load_grid(grid_calculation_function=uniform_grid)
print(grid_positions)
grid_B=load_field(field_generation_function=exact_ramp_field_grid, grid_positions=grid_positions)

#Generate a tree for indexing
mytree = scipy.spatial.cKDTree(grid_positions)

#############Set time############################33
B_magnitude = np.sqrt(np.sum(grid_B**2, axis=1))
print("Maximum field magnitude = " + str(np.max(B_magnitude)))
# exact_B_magnitude = np.sqrt(np.sum(grid_exact   **2, axis=1))
# print("Maximum exact field magnitude = " + str(np.max(exact_B_magnitude)))
dt_cyclotron = np.abs(0.1*2*np.pi/np.max(B_magnitude)/qmratio)
print("dt = " + str(dt))
print("dt cyclotron = " + str(dt_cyclotron))
dt=1e-13
seed=1
iters=N_iterations
colormaps={0.1:"Blues", 1.:"Greens", 10.:"Reds"}
# colormaps = {0.1:"Blues", 0.5:"Reds", 1.:"Greens",2.:"Oranges",10.:"Purples"}
for n in colormaps.keys():
    exact_path = particle_loop(pusher_function=RK4_step, field_calculation_function = exact_ramp_field,
        mode_name = "RK4_exact" + str(n), N_particles = N_particles, N_iterations=int(N_iterations*n),seed=seed,
        save_velocities=True, continue_run=False, dt=dt/n)
print("Finished calculation.")
display_quiver()
for n in colormaps.keys():
    display_particles(mode_name="RK4_exact" + str(n), colormap=colormaps[n])

print("Finished display")
mlab.show()
plot_energies("boris_exact", "RK4_exact")
