#!/bin/bash

import Scripts.imageForms as iF
import Scripts.hough as hg
import imageio
import cv2 as cv
import os
import sys

CURRENT_WORKING_DIRECTORY = os.getcwd()
VIDEO_PATHNAME = "\\Videos\\"
SCRIPTS_PATHNAME = "\\Scripts\\"
# @ Get currect working directory
print(CURRENT_WORKING_DIRECTORY)
# @ Add Video directory
sys.path.append(CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME)
sys.path.append(CURRENT_WORKING_DIRECTORY + SCRIPTS_PATHNAME)


if __name__ == "__main__":

    # TODO : Get a list of the files in the Videos Directory so i can turn this
    # TODO:  automatic and not write each file.
    # * Get the videos or images.
    FILENAME = "video-short(1).mp4"
    # hg.ShowVideo(CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME + FILENAME)
    vidCap = cv.VideoCapture(
        CURRENT_WORKING_DIRECTORY + VIDEO_PATHNAME + FILENAME)
    while True:
        ret, vidFrame = vidCap.read()
        if not ret:
            break
        videoGray = cv.cvtColor(vidFrame, cv.COLOR_BGR2GRAY)
        # videoCanny = cv.Canny(vidFrame, 100, 200)
        videoCanny = cv.Sobel(videoGray, cv.CV_8U, 2, 0)
        cv.imshow("Video", videoCanny)
        if cv.waitKey(20) >= 0:
            break
