from biot import *

#Simulation parameters
N_iterations=int(1e6)
Dump_every_N_iterations=N_iterations/100
N_particles=20
N_interpolation=8
velocity_scaling=1e6 #for random selection of initial velocity
seed=1
dt=1e-12
save_every_n_iterations=40
# particle_loop(pusher_function=RK4_step, field_calculation_function = test_z_field,
#     mode_name = "RK4", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
#     continue_run=False, dt=dt, preset_r=np.array([[0.008, 0., 0.]]), preset_v=np.array([[0,1000.,0]]),
#     save_every_n_iterations=50)
# particle_loop(pusher_function=boris_step, field_calculation_function = test_z_field,
#     mode_name = "boris", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
#     continue_run=False, dt=dt, preset_r=np.array([[0.008, 0., 0.]]), preset_v=np.array([[0,1000.,0]]),
#     save_every_n_iterations=50)




particle_loop(pusher_function=boris_step, field_calculation_function = exact_ramp_field,
    mode_name = "boris", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
    continue_run=False, dt=dt,
    save_every_n_iterations=save_every_n_iterations)

particle_loop(pusher_function=RK4_step, field_calculation_function = exact_ramp_field,
    mode_name = "RK4", N_particles = N_particles, N_iterations=int(N_iterations),seed=seed,
    continue_run=False, dt=dt,
    save_every_n_iterations=save_every_n_iterations)
plot_xy_positions(("boris", 'bo-'), ("RK4", 'ro-'))

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
