#!/bin/bash
import imageio
import cv2 as cv
import os
import sys
import numpy as np

CURRENT_WORKING_DIRECTORY = os.getcwd()
VIDEO_PATHNAME = "\\Videos\\"
IMAGES_PATHNAME = "\\images\\"
SCRIPTS_PATHNAME = "\\Scripts\\"
# @ Get currect working directory
print(CURRENT_WORKING_DIRECTORY)
# @ Add Video directory
sys.path.append(CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME)
sys.path.append(CURRENT_WORKING_DIRECTORY + SCRIPTS_PATHNAME)

from Scripts.gaussianBlur import GaussBlur
import Scripts.hough as hg
import Scripts.imageForms as iF
from Scripts.sobel import Sobel
from Scripts.gradient import Gradient
from Scripts.NonMaxSuppression import NonMaxSuppression
from Scripts.doubleThresholding import DoubleThresholding
from Scripts.valueMax import ImageMaxValue

from scipy import ndimage



if __name__ == "__main__":

    # TODO : Get a list of the files in the Videos Directory so i can turn this
    # TODO:  automatic and not write each file.
    # * Get the videos or images.
    FILENAME = "video-short(1).mp4"
    ImageFILENAME = "aula4-2.bmp"
    # ImageFILENAME = "peppers.jpg"
    # hg.ShowVideo(CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME + FILENAME)
    # test = gb.setup()

    image = cv.imread(CURRENT_WORKING_DIRECTORY +
                      IMAGES_PATHNAME + ImageFILENAME)
    
    # cv.imshow("original", image)
    # cv.waitKey(0 )
    # cv.imshow("window", videoGauss.resultImage)
    # cv.waitKey(0
    # cv.imshow("sks    ", videoSobel.resultImage)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    im = image.copy()
    # ! USE THE RED CHANNEL FOR BETTER PROCESSING OF EDGES AND ALL.
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
    maxValue = ImageMaxValue()
    # gaussImage = GaussFilter.GPUCalc(image)
    # gradientImage, theta = GradientFilter.GPUCalc(gaussImage)
    # maxSupImage =  MaxSup.GPUCalc(gradientImage, theta)
    # doubleT = doublt.GPUCalc(maxSupImage, 0.2,0.7)
    # 0.2 .6
    # cv.imshow("Gauss", gaussImage)
    # cv.imshow("gradient", gradientImage)
    # cv.imshow("MaxSup", maxSupImage)
    # cv.imshow("Double Threshold", doubleT)
    # cv.imshow("difference ", maxSupImage - doubleT)
    

    # cv.imshow("Double Thresh and Hysteresis", doubleT)
    # cv.imshow("Double Thresh and Hysteresis Diff",maxSupImage - doubleT)
    # cv.waitKey(0)
    vidCap = cv.VideoCapture(
        CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME + FILENAME)
    while True:
        ret, vidFrame = vidCap.read()
        if not ret:
            break
        # videoGray = cv.cvtColor(vidFrame, cv.COLOR_BGR2GRAY)
        # * Process images in the filters
    
        videoGauss = GaussFilter.GPUCalc(vidFrame)
        # VideoSobel = SobelFilter.GPUCalc(videoGauss)
        videoGrad, thetas = GradientFilter.GPUCalc(videoGauss)
        videonon = MaxSup.GPUCalc(videoGrad, thetas)
        videoT = doublt.GPUCalc(videonon, 0.4,0.8)
        # print(type(videoGauss.resultImage))
        # videoCanny = cv.Sobel(videoGray, cv.CV_8U, 2, 0)
        # videoGauss = gb.GPUCalc(videoGray, platform, device, ctx, commQ, prog,[[
        #                         1, 2, 1], [2, 4, 2], [1, 2, 1]], 9)

        cv.imshow("Video", videoT)
        if cv.waitKey(20) >= 0:
            break

