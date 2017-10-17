# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 11:21:25 2017

@author: Anirudh
"""

import numpy as np
import pickle
import cv2
import os

def LoadImages(folder):
    images =[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def Callibrate():
    calFolderName = "./camera_cal"
    images = LoadImages(calFolderName)
    nx = 9
    ny = 6
    #i = 1
    objPoints = []
    imgPoints = []
    objp = np.zeros((nx*ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
       
        if ret == True:
            imgPoints.append(corners)
            objPoints.append(objp)
            '''
            cv2.drawChessboardCorners(image, (nx, ny), corners, ret)
            cv2.imshow("image", image)
            cv2.imwrite("chessboard_image_{0}.jpeg".format(i), image)
            i = i+1
            cv2.waitKey()
            '''
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    
    coeff = mtx, dist
    
    pickle.dump(coeff, open("calibratioin.p", "wb"))

    return

if (__name__ == "__main__"):
    Callibrate()
