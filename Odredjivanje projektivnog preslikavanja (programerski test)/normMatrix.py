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
