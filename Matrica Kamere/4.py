import numpy as np
from numpy import linalg
import math
np.set_printoptions(precision=5, suppress=True) 
 
 # ovde pišete pomocne funkcije
 
def matricaKamere(pts2D, pts3D):
    A = []
    
    # Loop through the point correspondences
    for i in range(len(pts2D)):
        # Unpack 3D point (homogeneous coordinates)
        x, y, z, t = pts3D[i]
        
        # Unpack 2D point (homogeneous coordinates)
        u, v, w = pts2D[i]
        
        # The projection equation (2 equations for each point)
        A.append([0, 0, 0, 0, -w*x, -w*y, -w*z, -w*t, v*x, v*y, v*z, v*t])
        A.append([w*x, w*y, w*z, w*t, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u*t])
    
    A = np.array(A)
    
    # Perform SVD to solve for the camera matrix
    _, _, V = np.linalg.svd(A)
    
    # The solution to the equation is the last row of V (the smallest singular value)
    T = V[-1].reshape(3, 4)
    
    T = np.where(np.isclose(T, 0) , 0.0 , T)
    return T / T[-1][-1]
