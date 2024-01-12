#!/bin/bash
import math
import os
import sys

import cv2 as cv
import imageio
import numpy as np
from scipy import ndimage

CURRENT_WORKING_DIRECTORY = os.getcwd()
VIDEO_PATHNAME = "\\Videos\\"
IMAGES_PATHNAME = "\\images\\"
SCRIPTS_PATHNAME = "\\Scripts\\"
# @ Get currect working directory
print(CURRENT_WORKING_DIRECTORY)
# @ Add Video directory
sys.path.append(CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME)
sys.path.append(CURRENT_WORKING_DIRECTORY + SCRIPTS_PATHNAME)

import Scripts.imageForms as iF
from Scripts.doubleThresholding import DoubleThresholding
from Scripts.gaussianBlur import GaussBlur
from Scripts.gradient import Gradient
from Scripts.hough import HoughTransform
from Scripts.NonMaxSuppression import NonMaxSuppression
from Scripts.sobel import Sobel


if __name__ == "__main__":

    # TODO : Get a list of the files in the Videos Directory so i can turn this
    # TODO:  automatic and not write each file.
    # * Get the videos or images.
    FILENAME = "video-short(2).mp4"
    ImageFILENAME = "aula4-2.bmp"
    # ghouhFILENAME = "houghtest.bmp"
    # ghouhFILENAME = "aula4-3.bmp"
    # ImageFILENAME = "peppers.jpg"
    # hg.ShowVideo(CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME + FILENAME)
    # test = gb.setup()

    # image = cv.imread(CURRENT_WORKING_DIRECTORY +
    #                   IMAGES_PATHNAME + ImageFILENAME)
    # * Compare with sobel
    # imd = cv.cvtColor(im, cv.COLOR_BGR2BGRA)
    # sobbelx = cv.Sobel(im, cv.CV_64F,1,0, ksize=3)
    # sobbely = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)
    # grad = np.sqrt(sobbelx**2 + sobbely**2)
    # grad_norm = (grad * 255 / grad.max()).astype(np.uint8)
    # cv.imshow('s', grad_norm)
    # #####
    # * Initialize GPU calculated Filters and edge detection classes
    GaussFilter = GaussBlur(1.4)
    GradientFilter = Gradient()
    MaxSup = NonMaxSuppression()
    doublt = DoubleThresholding()
    # * Region of interest 
  
    hough = HoughTransform( minimumAngle=0,
       maximumAngle=180, angleSpacing=1, threshold=1.9)
    # gaussImage = GaussFilter.GPUCalc(image)
    # gradientImage, theta = GradientFilter.GPUCalc(gaussImage)
    # maxSupImage = MaxSup.GPUCalc(gradientImage, theta)
    # doubleT = doublt.GPUCalc(maxSupImage, 0.6, 0.3)
    # houghC = hough.GPUCalc(doubleT)
    
    # cv.imshow("Hough", houghC)
    # cv.waitKey(0)
    
    # @ Video handling
    vidCap = cv.VideoCapture(
        CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME + FILENAME)
    while True:
        ret, vidFrame = vidCap.read()

        if not ret:
            break
        # value_channel = videoGray[:,:,2]
        # * Process images in the filters
        # frameCopy = cv.resize(vidFrame, (vidFrame.shape[1]*0.30, vidFrame.shape[0]*0.30, 4), interpolation=cv.INTER_AREA)
        # pt1 = (vidFrame.shape[1]*0.30, vidFrame.shape[0]*0.30)
        # pt2 = (vidFrame.shape[1] - vidFrame.shape[1]*0.80,
        #        vidFrame.shape[0] - vidFrame.shape[0]*0.80)
        # print(pt2)

        # exit()
        # frameCopy = cv.rectangle(vidFrame.copy(),pt1,pt2,(255,0,0),2)
        # cv.se
        videoGauss = GaussFilter.GPUCalc(vidFrame.copy())
        videoGrad, thetas = GradientFilter.GPUCalc(videoGauss)
        videonon = MaxSup.GPUCalc(videoGrad, thetas)
        videoT = doublt.GPUCalc(videonon, 0.6, 0.2)
  
        # blank = np.zeros_like(videoT)
        # roi = np.array((videoT.shape[1] - videoT.shape[1]*0.80,
        #             videoT.shape[0] - videoT.shape[0]*0.80,4), dtype=np.int32)
        # roiP = cv.fillPoly(blank, roi, 255)
        # roiImage = cv.bitwise_and(videoT, roiP)

  
        videoHough, lines = hough.GPUCalc(videoT)
        for pos1, pos2 in lines:
            cv.line(vidFrame, pos1, pos2,(0,0,255), 1, cv.LINE_AA)
        cv.imshow("Video", vidFrame )
        if cv.waitKey(20) >= 0:
            break
