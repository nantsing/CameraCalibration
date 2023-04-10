### This is a reproduction of A compact algorithm for rectification of stereo pairs ###

import os
import numpy as np
import cv2 as cv

### Get P from camera parameters
def Get_P(A1, A2, R1, R2, t1, t2):

    Po1 = A1 @ np.column_stack((-R1, t1))
    Po2 = A2 @ np.column_stack((-R2, t2))

    return Po1, Po2

# ART: factorize a PPM as P = A*[R;t]
def art(P):
    Q = np.linalg.inv(P[:3, :3])
    U, B = np.linalg.qr(Q)
    
    R = np.linalg.inv(U)
    t = B @ P[:3, 3]
    A = np.linalg.inv(B)
    A = A / A[2, 2]
    
    return A, R, t

# RECTIFY: compute rectification matrices
def rectify(Po1, Po2, correction = 0):
    # factorize old PPMs
    A1, R1, t1 = art(Po1)
    A2, R2, t2 = art(Po2)
    
    #To keep the rectified image in the center of the window
    A1[0, 2] = A1[0, 2] + correction
    A2[0, 2] = A2[0, 2] + correction
    
    # optical centers (unchanged)
    c1 = - np.linalg.inv(Po1[:, :3]) @ Po1[:, 3]
    c2 = - np.linalg.inv(Po2[:, :3]) @ Po2[:, 3]
    
    # new x axis (= direction of the baseline)
    v1 = c1 - c2
    # new y axes (orthogonal to new x and old z)
    v2 = np.cross(R1[2, :].T, v1)
    # new z axes (orthogonal to baseline and y)
    v3 = np.cross(v1, v2)
    
    # new extrinsic parameters
    # translation is left unchanged
    R = np.array([v1.T / np.linalg.norm(v1, ord= 2), v2.T / np.linalg.norm(v2, ord= 2),\
        v3.T / np.linalg.norm(v3, ord= 2)])
    
    # new intrinsic parameters (arbitrary)
    A = (A1 + A2) / 2
    # no skew
    A[0, 1] = 0
    
    # new projection matrices
    Pn1 = A @ np.column_stack((R, (-R @ c1).T))
    Pn2 = A @ np.column_stack((R, (-R @ c2).T))
    
    # rectifying image transformation
    T1 = Pn1[:3, :3] @ np.linalg.inv(Po1[:3, :3])
    T2 = Pn2[:3, :3] @ np.linalg.inv(Po2[:3, :3])
    
    return T1, T2, Pn1, Pn2
    

if __name__ == '__main__':
    # P_old from paper
    Po1 = np.array([
        [9.765e2, 5.382e1, -2.398e2, 3.875e5],
        [9.849e1, 9.333e2, 1.574e2, 2.428e5],
        [5.790e-1, 1.108e-1, 8.077e-1, 1.118e3],
    ])
    Po2 = np.array([
        [9.767e2, 5.376e1, -2.400e2, 4.003e4],
        [9.868e1, 9.310e2, 1.567e2, 2.517e5],
        [5.766e-1, 1.141e-1, 8.089e-1, 1.174e3]
    ])
    
    T1, T2, Pn1, Pn2 = rectify(Po1, Po2, 160)
    
    image1 = cv.imread('./dataset/left.png')
    image2 = cv.imread('./dataset/right.png')
    shape = (image1.shape[1], image1.shape[0])
    print(shape)
    image1_rectified = cv.warpPerspective(image1, T1, shape, cv.INTER_LANCZOS4)
    image2_rectified = cv.warpPerspective(image2, T2, shape, cv.INTER_LANCZOS4)
    
    image_raw = np.hstack((image1, image2))
    image_raw = cv.line(image_raw, ( 0, shape[1] // 2 ), ( 2 * shape[0], shape[1] // 2 ), color=(255, 0, 0), thickness=1)
    cv.imwrite('./results/Images.png', image_raw)

    image_rectified = np.hstack((image1_rectified, image2_rectified))
    image_rectified = cv.line(image_rectified, ( 0, shape[1] // 2 ), ( 2 * shape[0], shape[1] // 2 ), color=(0, 0, 255), thickness=1)

    cv.imwrite('./results/RectifiedImages.png', image_rectified)