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


class GaussBlur():
    def __init__(self, sigma):
        # * Setup the program
        result = self.setup()
        # print(result)
        # * Get the kernel size out of the sigma
        self.kernel_size = 2 * int(3 * sigma) + 1
        self.gaussian_kernel = self.create_gaussian_kernel(sigma, self.kernel_size)
        self.gaussian_kernel = self.gaussian_kernel / np.sum(self.gaussian_kernel)

        # self.resultImage = self.GPUCalc(image, sigma)

    @property
    def image_result(self):
        return self.resultImage

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
            _file = open(os.getcwd() + "\\Scripts\\gaussianBlur.cl", "r")
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

    def create_gaussian_kernel(self, sigma, size):
        kernel = np.zeros(size, dtype=np.float32)
        center = size // 2
        for i in range(size):
            kernel[i] = np.exp(-((i - center) / (2 * sigma)) ** 2)
        return kernel / np.sum(kernel)

    def GPUCalc(self, image:np.ndarray):
        # TODO: Check if i really want the image to come ou t black and white
        # ! The Image is processed but it comes out black and white. 
        try:
            # * Convert Image to BGRA
            imageBGRA = cv.cvtColor(image,  cv.COLOR_BGR2BGRA)


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
          
            # print(gaussian_kernel)
            
            # * Mask Buffer
            maskBuffer = cl.Buffer(
                ctx,
                flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                hostbuf=self.gaussian_kernel
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
            
            # * Setup local and global working memory
            localws = (32, 16)  # openCV 32x8 = 256
            globalws = (math.ceil(width / localws[0]) * localws[0],
                        math.ceil(height / localws[1]) * localws[1])
            # * Send Paramenters to device
            kernelName = prog.gaussianBLUR_image2D
      

            kernelName.set_arg(0, imageIn)
            kernelName.set_arg(1, imageOut)
            kernelName.set_arg(2, np.int32(self.kernel_size))
            kernelName.set_arg(3, maskBuffer)
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
           
            maskBuffer.release()
            imageIn.release()
            imageOut.release()
            return imageBGRA
        except Exception as e:
            # TODO: Attach logger
            print(e)
