import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True)

def normMatrix(src_p):

    # teziste sistema tacaka C(x, y)
    x = sum([p[0]/p[2] for p in src_p]) / len(src_p)
    y = sum([p[1]/p[2] for p in src_p]) / len(src_p)

    # srednje rastojanje
    r = 0.0

    for i in range(len(src_p)):
        # translacija u koordinatni pocetak
        tmp1 = float(src_p[i][0]/src_p[i][2]) - x
        tmp2 = float(src_p[i][1]/src_p[i][2]) - y

        r = r + math.sqrt(tmp1**2 + tmp2**2)

    r = r / float(len(src_p))

    # skaliranje
    S = float(math.sqrt(2)) / r

    # vracamo matricu normalizacije
    return np.array([[S, 0, -S*x], [0, S, -S*y], [0, 0, 1]])

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

def DLTwithNormalization(src_p, dst_p):

    # transformacije
    T = normMatrix(src_p)
    T_prim = normMatrix(dst_p)

    # normalizovane tacke
    M_line = T.dot(np.transpose(src_p))
    M_prim = T_prim.dot(np.transpose(dst_p))

    M_line = np.transpose(M_line)
    M_prim = np.transpose(M_prim)

    P_line = DLT(M_line, M_prim)

    P = (np.linalg.inv(T_prim)).dot(P_line).dot(T)

    if(P[2][2] != 1 and P[2][2] != 0):
        P = P/P[2][2]

    return P

