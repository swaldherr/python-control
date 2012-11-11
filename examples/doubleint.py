# doubleint.py - double integrator example
# RMM, 10 Nov 2012
#
# This example shows how to compute a trajectory for a very simple double
# integrator system.  Mainly useful to show the simplest type of trajectory
# generation computation.

import control as ctrl                 # control system toolbox
import trajgen as tg          # trajectory generation toolbox

# Define a double integrator system
sys1 = ctrl.tf2ss(ctrl.tf([1], [1, 0, 0]))

# Set the initial and final conditions
x0 = (0, 0);
xf = (1, 1);

# Find a trajectory
xd, ud = tg.linear_point_to_point(sys1, x0, xf, 1)

# Plot the trajectory
