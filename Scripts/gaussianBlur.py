import sys
import cv2 as cv
import GPL_LIB as gpl
import numpy
import PIL import Image
import pyopencl as cl
import os 


def setup():
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
        file = open("gaussianBlur.cl", "r")
        # * Get the Kernel
        global prog
        prog = cl.Program(ctx, file.read())
        # * Build the Kernel Program
        prog.buils()
        
    except Exception as e:
        # TODO: Attach a logger.
        print(e)

    return platform, device, ctx, commQ, prog

def GPUCalc(image, platform, device, ctx, commQ, prog, kernelMask, KernelSize):
    try:
        imageBGRA = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        imgCopy= imageBGRA.copy()
        
        imgFormat = cl.ImageFormat(
            cl.channel_order.BGRA,
            cl.channel_type.UNSIGNED_INT8
        )
        
        # * Buffer In
        bufferIn = cl.Image()
        
        # * Buffer Out
        
    except Exception as e:
        # TODO: Attach logger 
        print(e)
        