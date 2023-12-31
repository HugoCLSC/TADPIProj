import cv2 as cv 
import numpy as np
from matplotlib import pyplot as plt
import imageForms as iF
import math

def ShowVideo(filename):
    """
    Play a video with OpenCV
    :param filename:
    :return:
    """
    vidCap = cv.VideoCapture(filename)

    if (not vidCap.isOpened()):
        print("Video File Not Found")
        exit(-1)

    while (True):
        ret, vidFrame = vidCap.read()
        if (not ret):
            break

        # TODO image processing for showing Hough Lines
        # videoCanny = cv.Canny(vidFrame, 100, 200)
        # videoLines = ShowHoughLines(videoCanny, vidFrame, 150)
        # videoLines = ShowHoughLineSegments(videoCanny, vidFrame, 200)

        # cv.imshow("Video", videoLines)
        cv.imshow("Video", vidFrame)

        if (cv.waitKey(20) >= 0):  # Press q to exit the video
            break
