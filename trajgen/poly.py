# poly.m - simple set of polynomial basis functions
# RMM, 10 Nov 2012
#
# This class implements a set of simple basis functions consisting of powers
# of t: 1, t, t^2, ...

import numpy as np
import scipy as sp

class Poly:
    def __init__(self, N):
        self.N = N                    # save number of basis functions

    # Compute the kth derivative of the ith basis function at time t
    def eval_deriv(self, i, k, t):
        if (i < k): return 0;           # higher derivative than power
        return sp.misc.factorial(i)/sp.misc.factorial(i-k) * np.power(t, i-k)
