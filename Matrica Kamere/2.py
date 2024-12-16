import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True)

 # ovde pišete pomocne funkcije

def kameraK(T):

 # vaš kod
    T0 = T[:, :-1]
    Q, R = linalg.qr(linalg.inv(T0))

    if R[0][0] < 0:
        R[0] = -R[0]
        Q[:,0] = -Q[:,0]
    if R[1][1] < 0:
        R[1] = -R[1]
        Q[:,1] = -Q[:,1]
    if R[2][2] < 0:
        R[2] = -R[2]
        Q[:,2] = -Q[:,2]

    K = T0.dot(Q)
    if K[2][2] != 1:
        K = K / K[2][2]
    K = np.where(np.isclose(K, 0) , 0.0 , K)
    return K
