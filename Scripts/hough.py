import sys
import cv2 as cv
import numpy as np
from PIL import Image
import pyopencl as cl
import imageForms as iF
import os
import sys
import math
sys.path.append(os.getcwd() + "\\Scripts\\")


class HoughTransform ():
    def __init__(self):
        # * Setup the program
        result = self.setup()

    def setup(self):
        try:
            # * Configure the platform
            platforms = cl.get_platforms()
            global platform
            platform = platforms[0]
            # * Configure Devices
            devices = platform.get_devices()
            global device
            device = devices[0]
            # * Set context
            global ctx
            ctx = cl.Context(devices)
            # * Create Command Queue
            global commQ
            commQ = cl.CommandQueue(ctx, device)
            # * Load Gaussian kernel file
            _file = open(os.getcwd() + "\\Scripts\\Hough.cl", "r")
            # * Kernel build options

            # * Get the Kernel
            global prog
            prog = cl.Program(ctx, _file.read())
            # * Build the Kernel Program
            prog.build()
        except Exception as e:
            # TODO: Attach a logger.
            print(e)
            return False

        return True

    def GPUCalc(self, image: np.ndarray):
        pass
