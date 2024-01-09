import sys
import cv2 as cv 
import numpy as np
from PIL import Image
import pyopencl as cl
import imageForms as iF
import os 
import math 

sys.path.append(os.getcwd() + "\\Scripts\\")

class Gradient():
    
    def __init__(self):
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
            _file = open(os.getcwd() + "\\Scripts\\gradient.cl", "r")
            # * Kernel build options

            # * Get the Kernel
            global prog
            prog = cl.Program(ctx, _file.read())
            # * Build the Kernel Program
            prog.build()
        except Exception as e:
            # TODO: Attach a logger 
            print(e)
            return False
        return True
    
    def GPUCalc(seld, image:np.ndarray):
        try:
            # * Convert Image to BGRA

            # imageBGRA = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            imageBGRA = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
            # imgCopy = imageBGRA.copy()

            # * Get image Properties
            height = imageBGRA.shape[0]
            width = imageBGRA.shape[1]
            widthStep = imageBGRA.strides[0]
            nChannels = imageBGRA.shape[2]
            padding = imageBGRA.strides[0] - width * \
                imageBGRA.strides[1] * imageBGRA.itemsize

            imgFormat = cl.ImageFormat(
                cl.channel_order.BGRA,
                cl.channel_type.UNSIGNED_INT8
            )

            # * IMAGE Buffer In
            imageIn = cl.Image(
                ctx,
                flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                format=imgFormat,
                shape=(imageBGRA.shape[1], imageBGRA.shape[0]),
                pitches=(imageBGRA.strides[0], imageBGRA.strides[1]),
                hostbuf=imageBGRA.data
            )
            # * IMAGE Buffer Out
            imageOut = cl.Image(
                ctx,
                flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.WRITE_ONLY,
                format=imgFormat,
                shape=(imageBGRA.shape[1], imageBGRA.shape[0]),
                pitches=(imageBGRA.strides[0], imageBGRA.strides[1]),
                hostbuf=imageBGRA.data
                
            )
            # np_result = np.array(np.zeros((width, height, 4)), dtype=np.double)
            np_result = np.array(np.zeros((width, height)), dtype=np.double)
            # The np_result will have a matrix with the width and height equal to the image
            bufferResult = cl.Buffer(
                ctx,
                flags=cl.mem_flags.USE_HOST_PTR | cl.mem_flags.WRITE_ONLY,
                size=np_result.nbytes,

                hostbuf=np_result
            )
            # * Setup work and group items
            localws = (32, 16)  # openCV 32x8 = 256
            globalws = (math.ceil(width / localws[0]) * localws[0],
                        math.ceil(height / localws[1]) * localws[1])
            # * Send Paramenters to device
            kernelName = prog.gradient_image2D

            kernelName.set_arg(0, imageIn)
            kernelName.set_arg(1, imageOut)
            kernelName.set_arg(2, np.int32(width))
            kernelName.set_arg(3, np.int32(height))
            kernelName.set_arg(4, bufferResult)

            # * Start the kernel program
            kernelEvent = cl.enqueue_nd_range_kernel(commQ,
                                                     kernelName,
                                                     global_work_size=globalws,
                                                     local_work_size=localws
                                                     ).wait()

            # * Get result form the program
            cl.enqueue_copy(
                commQ,
                dest=imageBGRA,
                src=imageOut,
                origin=(0, 0, 0),
                region=(imageBGRA.shape[1], imageBGRA.shape[0]),
                is_blocking=True
            )
            cl.enqueue_copy(
                commQ,
                dest=np_result,
                src=bufferResult,
                is_blocking=True
            )
            # print(np_result)
            bufferResult.release()
            imageIn.release()
            imageOut.release()
            return imageBGRA, np_result
        except Exception as e:
            print(e)
