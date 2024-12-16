import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True)

# ovde pišete pomocne funkcije

def kameraA(T):
    # vaš kod
    T0 = T[:, :-1]
    Q, R = linalg.qr(linalg.inv(T0))

    # Ensure R's diagonal is positive, and adjust Q accordingly
    for i in range(min(R.shape[0], R.shape[1])):
        if R[i, i] < 0:
            R[i] = -R[i]
            Q[:, i] = -Q[:, i]

    # Check and adjust the sign of Q to match a consistent convention
    if np.linalg.det(Q) < 0:
        Q = -Q

    A = Q

    # Ensure numerical stability by zeroing near-zero elements
    A = np.where(np.isclose(A, 0), 0.0, A)
    return A

