from biot import *

display_wires(N_wires=6, r_wires=r_wires)
# display_quiver(field_mode_name="biot")
display_particles(mode_name="boris", colormap="Blues")
display_particles(mode_name="RK4", colormap="Reds")
print("Finished display")
mlab.show()
