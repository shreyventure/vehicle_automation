from typing import AnyStr, List
import os
import matplotlib.image as mpimg
import numpy as np
import cv2

def calibrate_camera(directory: AnyStr):
    '''
    @params:
    directory: The path to the directory where chessboard images are stored

    @return: (a tuple)
    retval, cameraMatrix, distCoeffs, rvecs, tvecs 
    '''
    cal_img_paths = os.listdir(directory)
    for i in range(len(cal_img_paths)):
        cal_img_paths[i] = directory +'/' + cal_img_paths[i]

    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    nx, ny = 9, 6
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    for i in range(len(cal_img_paths)):
        image = mpimg.imread(cal_img_paths[i])
        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points and image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs
