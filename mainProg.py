# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:47:46 2017

@author: Anirudh
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def LoadImages(folder):
    images =[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def Debug():
    imagesDebug = LoadImages("./test_images")
    mtx, dist = Callibrate(LoadImages("camera_cal"))
    plt.imshow(imagesDebug [0])
    plt.show()
    return

def Sharpen(image):
    gb = cv2.GaussianBlur(image, (5,5), 20.0)
    return cv2.addWeighted(image, 2, gb, -1, 0)


def Callibrate(images):
    nx = 9
    ny = 6
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
            
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)

    return mtx, dist

def UnDist(image , mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)
    

def GetTransform():
    src = np.float32([[0, 673], [1207, 673], [0, 450], [1280, 450]])
    dst = np.float32([[569, 223], [711, 223], [0, 0], [1280, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M , Minv

def Transform(image , M, img_size):
    return cv2.warpPerspective(image, M, img_size)
    

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return sxbinary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    mag = ((sobelx)**2+(sobely)**2)**0.5
    scaled_sobel = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dir)
    binary_output[(dir >= thresh[0]) & (dir <= thresh[1])] = 1
    return binary_output
    
def GradientThresholdImage(image, thresholds = (20, 100)):
    ksize = 3
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=thresholds)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=thresholds)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=thresholds)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
    color_binary = ColorThresholdImage(image, thresholds = (190, 255))
    combined = np.zeros_like(color_binary)
    #combined[(((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))) | color_binary == 1] = 1
    combined = color_binary
    return combined
    

def ColorThresholdImage(image, thresholds = (0, 255)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    b_channel = lab[:,:, 2]
    l_channel = l_channel*(255/np.max(l_channel))
    if (np.max(b_channel) > 175):
        b_channel = b_channel*(255/np.max(b_channel))
    binary_output = np.zeros_like(l_channel)
    binary_output[((l_channel > 220) & (l_channel <= 255)) | ((b_channel > thresholds[0]) & (b_channel <= thresholds[1]))  ] = 1
    return binary_output

def ThresholdImage(image):
    stage1 = GradientThresholdImage(image)
    stage2 = ColorThresholdImage(image)
    return

def PlotImages(images):
    i = 1
    for image in images:
        cv2.imshow("image {0}".format(i), image)
        cv2.waitKey()
        i = i+1
    return

def PipeLine():
    calFolderName = "./camera_cal"
    images = LoadImages(calFolderName)
    mtx, dist = Callibrate(images)
    
    testFolderName = "./test_images"
    imagesTest = LoadImages(testFolderName)
    
    img_size = (1280, 223)
    
    undistortedTestImages = []
    
    M, Minv = GetTransform()
    
    for image in imagesTest:
        undist = UnDist(image, mtx, dist)
        #undist = Sharpen(undist)
        undist = GradientThresholdImage(undist)
        undist = Transform(undist, M, img_size)
        
        undistortedTestImages.append(undist)
        
    PlotImages(undistortedTestImages)
    return
    

if (__name__ == "__main__"):
    PipeLine()
    #Debug()
