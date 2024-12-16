import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True)

 # ovde pišete pomocne funkcije

def centar(T):

 # vaš kod

    T0 = T[:, :-1]

    if linalg.det(T0) < 0:
        T = -T

    # -- C --------------------------------------------
    c1 = linalg.det(T[:, 1:])
    c2 = -linalg.det(np.array([T[:,0], T[:,2], T[:,3]]).T)
    c3 = linalg.det(np.array([T[:,0], T[:,1], T[:,3]]).T)
    c4 = -linalg.det(T[:, :-1])

    c1 = c1/c4
    c2 = c2/c4
    c3 = c3/c4

    C = np.array([c1, c2, c3, 1])
    C = np.where(np.isclose(C, 0) , 0.0 , C)
    return C
