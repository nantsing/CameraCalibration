import cv2 as cv
import numpy as np
import rectification

from rectify import *
from Transform import *

if __name__ == "__main__":
    # Left image
    img1 = cv.imread("dataset/1.png")
    # Right image       
    img2 = cv.imread("dataset/2.png")     
    dims1 = (img1.shape[1], img1.shape[0])               
    dims2 = (img2.shape[1], img2.shape[0])
    image = np.hstack((img1, img2))
    cv.imwrite('./results/DirImages.png', image)

    # Distortion coefficients
    distCoeffs1 = np.array([])   
    distCoeffs2 = np.array([])

    # Left camera intrinsic matrix
    A1 = np.array([[ 960, 0, 960/2], [0, 960, 540/2], [0,0,1]])
    # Right camera intrinsic matrix
    A2 = np.array([[ 960, 0, 960/2], [0, 960, 540/2], [0,0,1]])
    # Left extrinsic parameters             
    RT1 = np.array([[ 0.98920029, -0.11784191, -0.08715574,  2.26296163],   
                    [-0.1284277 , -0.41030705, -0.90285909,  0.15825593],
                    [ 0.07063401,  0.90430164, -0.42101002, 11.0683527 ]])
    # Right extrinsic parameters
    RT2 = np.array([[ 0.94090474,  0.33686835,  0.03489951,  1.0174818 ],   
                    [ 0.14616159, -0.31095025, -0.93912017,  2.36511779],
                    [-0.30550784,  0.88872361, -0.34181178, 14.08488464]])
    
    # 3x4 camera projection matrices
    Po1 = A1 @ RT1                               
    Po2 = A2 @ RT2
    
    # Fundamental matrix F 
    F = rectification.getFundamentalMatrixFromProjections(Po1, Po2)
    
    Rectify1, Rectify2 = rectification.getDirectRectifications(A1, A2, RT1, RT2, dims1, dims2, F)
    destDims = dims1
    
    # Get fitting affine transformation to fit the images into the frame
    Fit = rectification.getFittingMatrix(A1, A2, Rectify1, Rectify2, dims1, dims2, distCoeffs1, distCoeffs2)
    
    # Compute maps with OpenCV considering rectifications, fitting transformations and lens distortion
    mapx1, mapy1 = cv.initUndistortRectifyMap(A1, distCoeffs1, Rectify1.dot(A1), Fit, destDims, cv.CV_32FC1)
    mapx2, mapy2 = cv.initUndistortRectifyMap(A2, distCoeffs2, Rectify2.dot(A2), Fit, destDims, cv.CV_32FC1)
    
    # Apply final transformation to images 
    img1_rect = cv.remap(img1, mapx1, mapy1, interpolation=cv.INTER_LINEAR)
    img2_rect = cv.remap(img2, mapx2, mapy2, interpolation=cv.INTER_LINEAR)
    
    # Visualise as single image
    rectImgs = np.hstack((img1_rect, img2_rect))
    rectImgs = cv.line(rectImgs, ( 0, destDims[1] // 2 ), ( 2 * destDims[0], destDims[1] // 2 ), color=(0,0,255), thickness=1)
    
    # Show images
    cv.imwrite('./results/DirRectifiedImages.png', rectImgs)