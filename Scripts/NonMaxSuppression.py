import sys
import cv2 as cv 
import numpy as np
from PIL import Image
import pyopencl as cl
import imageForms as iF
import os
import math

sys.path.append(os.getcwd() + "\\Scripts\\")


class NonMaxSuppression():

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
            _file = open(os.getcwd() + "\\Scripts\\NonMaxSuppression.cl", "r")
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

    def GPUCalc(self, image: np.ndarray, thetaMatrix):
        try:
            # print(thetaMatrix)
            # * Convert Image to BGRA
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
            thetaBuffer = cl.Buffer(
                ctx, 
                flags = cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                hostbuf = thetaMatrix
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
            # * Setup work and group items
            localws = (32, 16)  # openCV 32x8 = 256
            globalws = (math.ceil(width / localws[0]) * localws[0],
                        math.ceil(height / localws[1]) * localws[1])
            # * Send Paramenters to device
            kernelName = prog.non_max_suppression_image2D

            # # * Sobel filter for horizontal and vertical directions

            kernelName.set_arg(0, imageIn)
            kernelName.set_arg(1, imageOut)
            kernelName.set_arg(2, np.int32(padding))
            kernelName.set_arg(3, thetaBuffer)
            kernelName.set_arg(4, np.int32(width))
            kernelName.set_arg(5, np.int32(height))

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
            thetaBuffer.release()
            imageIn.release()
            imageOut.release()
            
            return imageBGRA
        except Exception as e:
            print(e)
