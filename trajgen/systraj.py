import numpy as np

class SystemTrajectory:
    def __init__(self, states, inputs):
        self.states = states
        self.inputs = inputs

    # Evaluate the trajectory over a list of time points
    def eval(self, tlist):
        # Allocate space for the outputs
        xd = np.zeros((len(tlist), self.states))
        ud = np.zeros((len(tlist), self.inputs))

        # Go through each time point and compute xd and ud via flat variables
        for k in range(len(tlist)):
            zflag = np.zeros(self.states + self.inputs)
            for i in range(self.states + self.inputs):
                for j in range(self.basis.N):
                    #! TODO: rewrite eval_deriv to take in time vector
                    zflag[i] += self.coeffs[j] * \
                        self.basis.eval_deriv(j, i, tlist[k])

            # Now copy the states and inputs
            xd[k,:], ud[k,:] = self.system.reverse(zflag)

        return xd, ud

