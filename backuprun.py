from biot import *

grid_positions, dx, dy, dz=load_grid(grid_calculation_function=uniform_grid)
print(grid_positions)
grid_B=load_field(field_generation_function=exact_ramp_field_grid)

#Simulation parameters
N_iterations=int(1e8)
Dump_every_N_iterations=int(1e6)
N_particles=10
N_interpolation=8
velocity_scaling=1e6 #for random selection of initial velocity
dt=0.01/velocity_scaling

#####Comparing fields calculated via exact result and biot savart for single wire case
# grid_exact = exact_single_wire_field_grid(N=N)
# grid_exact = load_field(field_generation_function=exact_single_wire_field_grid)
# scales=[]
# N_list=[5, 10, 15, 20, 25, 30, 40, 50, 60, 100, 125, 150, 175, 200, 300, 400, 500, 600, 700, 750, 800, 900, 1000]
# for N in N_list:
#     print N
#     scales.append(display_difference_quiver(grid_exact, biot_savart_field(N=N, N_wires = 1, r_wires=0.)))
# display_difference_quiver(grid_B, grid_exact)
# print("Biot savart - niebieskie, exact - czerwone")
# plt.plot(N_list, scales)
# plt.show()

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
dt = dt_cyclotron
dt=1e-13
seed=4
iters=N_iterations
N_particles=1


# exact_path = particle_loop(pusher_function=boris_step, field_calculation_function = exact_ramp_field,
#     mode_name = "boris_exact", N_particles = N_particles, N_iterations=iters,seed=seed, save_velocities=False)
exact_RK4_path = particle_loop(pusher_function=RK4_step, field_calculation_function = exact_ramp_field,
    mode_name = "RK4_exact", N_particles = N_particles, N_iterations=iters,seed=seed, save_velocities=True)
# N_interpolation_list=range(2,50)
# variances=[]
# for N_interpolation in N_interpolation_list:
# test_path=particle_loop(pusher_function=RK4_step, field_calculation_function = exact_ramp_field,
#         mode_name = "RK4_exact", N_particles = N_particles, N_iterations=iters,seed=seed,
#         N_interpolation=N_interpolation, save_velocities=True)
    # variance = compare_trajectories(exact_path, test_path)
    # variances.append(variance)
print("Finished calculation.")
# plt.plot(N_interpolation_list, variances)
# plt.show()
# compare_trajectories(exact_path,test_path)
# display_wires(N_wires=1, r_wires=0)
display_quiver()
display_particles(mode_name="boris_exact", colormap="Blues")
display_particles(mode_name="RK4_exact", colormap="Reds")

# print("Finished display")
mlab.show()
# plot_energies("boris_exact", "RK4_exact")
