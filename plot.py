from biot import *

# display_wires(N_wires=6, r_wires=r_wires)
display_wires(N_wires=6, r_wires=r_wires)
display_quiver()
display_particles(mode_name="boris", colormap="Blues")
# display_particles(mode_name="borisexact", colormap="Blues")
print("Finished display")
mlab.show()
