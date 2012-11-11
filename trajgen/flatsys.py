# flatsys.py - trajectory generation for differentially flat systems
# RMM, 10 Nov 2012
#
# This file contains routines for computing trajectories for differentially
# flat nonlinear systems.  It is (very) loosely based on the NTG software
# package developed by Mark Milam and Kudah Mushambi, but rewritten from
# scratch in python.

import numpy as np
import control
from trajgen import Poly

# Solve a point to point trajectory generation problem for a linear system
def linear_point_to_point(sys, x0, xf, Tf, basis=None, cost=None, T0 = 0):
    #
    # Make sure the probelm is one that we can handle
    #
    if (not control.isctime(sys)):
        raise control.ControlNotImplemented(
            "requires continuous time, linear control system")
    elif (not control.issiso(sys)):
        raise control.ControlNotImplemented(
            "only single input, single output systems are supported")

    #
    # Determine the basis function set to use and make sure it is big enough
    #

    # If no basis set was specified, use a polynomial basis (poor choice...)
    if (basis is None): basis = Poly(2*sys.states)
    
    # Make sure we have enough basis functions to solve the problem
    if (basis.N < 2*sys.states):
        raise ValueError("basis set is too small")

    #
    # Find the transformation to bring the system into reachable form
    # and use this to determine the flat output variable z = Cf*x
    #
    zsys, Tr = control.reachable_form(sys)
    Cfz = np.zeros(np.shape(sys.C)); Cfz[-1] = 1
    Cfx = Cfz * Tr

    #
    # Map the initial and final conditions to flat output conditions
    #
    # We need to compute the output "flag": [z(t), z'(t), z''(t), ...]
    # and then evaluate this at the initial and final condition.
    #
    zflag_T0 = np.zeros((sys.states, 1));
    zflag_Tf = np.zeros((sys.states, 1));
    H = Cfx                             # initial state transformation
    for i in range(sys.states):
        zflag_T0[i, 0] = H * np.matrix(x0).T
        zflag_Tf[i, 0] = H * np.matrix(xf).T
        H = H * sys.A                   # derivative for next iteration

    #
    # Compute the matrix constraints for initial and final conditions
    #
    # This computation depends on the basis function we are using.  It
    # essentially amounts to evaluating the basis functions and their
    # derivatives at the initial and final conditions.

    # Start by creating an empty matrix that we can fill up
    M = np.zeros((2*sys.states, basis.N))

    # Now fill in the rows for the initial and final states
    for i in range(sys.states):
        for j in range(basis.N):
            M[i, j] = basis.eval_deriv(j, i, T0)
            M[sys.states + i, j] = basis.eval_deriv(j, i, Tf)

    #
    # Solve for the coefficients of the flat outputs
    #
    # At this point, we need to solve the equation M alpha = zflag, where M
    # is the matrix constrains for initial and final conditions and zflag =
    # [zflag_T0; zflag_tf].  Since everything is linear, just compute the
    # least squares solution for now.
    #
    #! TODO: need to allow cost and constraints...
    alpha = np.linalg.pinv(M) * np.vstack((zflag_T0, zflag_Tf))

    #
    # Transform the trajectory from flat outputs to states and inputs
    #
    xdfcn = alpha
    udfcn = lambda t: 0

    # Return a function that computes inputs and states as a function of time
    return xdfcn, udfcn
