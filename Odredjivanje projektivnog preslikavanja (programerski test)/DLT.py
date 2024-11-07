import numpy as np
from numpy import linalg
np.set_printoptions(precision=5, suppress=True)

def DLT(src_p, dst_p):
    x = src_p[0][0]
    y = src_p[0][1]
    z = src_p[0][2]

    u = dst_p[0][0]
    v = dst_p[0][1]
    w = dst_p[0][2]

    A = np.array([
        [0, 0, 0, -w*x, -w*y, -w*z, v*x, v*y, v*z],
        [w*x, w*y, w*z, 0, 0, 0, -u*x, -u*y, -u*z]
    ])

    for i in range(1, len(src_p)):
        x = src_p[i][0]
        y = src_p[i][1]
        z = src_p[i][2]

        u = dst_p[i][0]
        v = dst_p[i][1]
        w = dst_p[i][2]

        row1 = np.array([0, 0, 0, -w*x, -w*y, -w*z, v*x, v*y, v*z])
        row2 = np.array([w*x, w*y, w*z, 0, 0, 0, -u*x, -u*y, -u*z])

        A = np.vstack((A, row1))
        A = np.vstack((A, row2))

    # print(A.shape)
    # print(A)

    # SVD dekompozicija
    U, S, V = np.linalg.svd(A)

    P = V[-1].reshape(3,3)

    if(P[2][2] != 1 and P[2][2] != 0):
        P = P/P[2][2]

    return P
