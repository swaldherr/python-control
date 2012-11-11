import numpy as np
import control

class LinearFlatSystem:
    def __init__(self, sys):
        self.A = sys.A
        self.B = sys.B

        # Find the transformation to bring the system into reachable form
        zsys, Tr = control.reachable_form(sys)
        self.F = zsys.A[0,:]            # input function coeffs
        self.T = Tr                     # state space transformation
        self.Tinv = np.linalg.inv(Tr)   # computer inverse once
        
        # Compute the flat output variable z = C x
        Cfz = np.zeros(np.shape(sys.C)); Cfz[0, -1] = 1
        self.C = Cfz * Tr

        # Keep track of the number of states and inputs
        self.states = sys.states
        self.inputs = sys.inputs

    # Compute state and input from flat flag
    def reverse(self, zflag):
        x = self.Tinv * np.matrix(zflag[-2::-1]).T
        u = zflag[-1] - self.F * x
        return np.reshape(x, self.states), np.reshape(u, self.inputs)
