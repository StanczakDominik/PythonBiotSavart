from biot import *

#Simulation parameters
N_iterations=int(1.5e6)
Dump_every_N_iterations=N_iterations/100
N_particles=10
N_interpolation=8
velocity_scaling=1e4 #for random selection of initial velocity
preset_z_velocity=1e4
seed=1
dt=1e-9
save_every_n_iterations=1


# mytree = scipy.spatial.cKDTree(grid_positions)

# particle_loop(pusher_function=RK4_step, field_calculation_function = test_z_field,
#     mode_name = "RK4", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
#     continue_run=False, dt=dt, preset_r=np.array([[0.008, 0., 0.]]), preset_v=np.array([[0,1000.,0]]),
#     save_every_n_iterations=50)
# particle_loop(pusher_function=boris_step, field_calculation_function = test_z_field,
#     mode_name = "boris", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
#     continue_run=False, dt=dt, preset_r=np.array([[0.008, 0., 0.]]), preset_v=np.array([[0,1000.,0]]),
#     save_every_n_iterations=50)

# from time import time

particle_loop(pusher_function=boris_step, field_calculation_function = exact_ramp_field,
    mode_name = "boris", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
    continue_run=False, dt=dt,
    save_every_n_iterations=save_every_n_iterations,
    position_sampler=bottom_position_sampler, velocity_sampler = directional_velocity_sampler,
    preset_z_velocity=preset_z_velocity, velocity_scaling=velocity_scaling)
# particle_loop(pusher_function=RK4_step, field_calculation_function = exact_ramp_field,
#     mode_name = "RK4", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
#     continue_run=False, dt=dt,
#     save_every_n_iterations=save_every_n_iterations,
#     position_sampler=bottom_position_sampler, velocity_sampler = directional_velocity_sampler,
#     preset_z_velocity=preset_z_velocity, velocity_scaling=velocity_scaling)
# # particle_loop(pusher_function=boris_step, field_calculation_function = exact_ramp_field,
# #     mode_name = "boris_smooth", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
# #     continue_run=False, dt=dt,
# #     save_every_n_iterations=1)

# test_exact_ramp_field()
# grid_positions, dx, dy, dz=load_grid(grid_calculation_function=uniform_grid)
# grid_B=load_field(field_generation_function=exact_ramp_field_grid, grid_positions=grid_positions)

# display_quiver()
# display_particles(("boris", "Blues"))#, ("RK4", "Reds"))
# print("Boris runtime %f, RK4 runtime %f"%(boris_runtime, rk4_runtime))
# mlab.show()
#
# plot_energies(("boris", "b-"), ("RK4", "r-"))

# print("Finished calculation.")
# RK4_x = RK4_path[:,0]
# RK4_y = RK4_path[:,1]
# boris_x = boris_path[:,0]
# boris_y = boris_path[:,1]

# plt.plot(RK4_x, RK4_y, "ro-", label="RK4")
# plt.plot(boris_x, boris_y, "bo-", label="Boris")
# plt.grid()
# plt.legend()
# plt.show()
# print("Finished display")
# plot_energies("boris", "RK4")
