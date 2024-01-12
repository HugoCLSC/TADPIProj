import sys
import cv2 as cv
import numpy as np
from PIL import Image
import pyopencl as cl
import imageForms as iF
import os
import sys
import math
from sklearn.cluster import DBSCAN
np.set_printoptions(suppress=True, linewidth=sys.maxsize,
                    threshold=sys.maxsize)

sys.path.append(os.getcwd() + "\\Scripts\\")

# np.set_printoptions(suppress=True, linewidth=sys.maxsize,
#                     threshold=sys.maxsize)
sys.stdout = open('output.txt', 'w')
# sys.stdout.close()
# sys.stdout = open('output.txt', 'w')


class HoughTransform ():
    def __init__(self,  minimumAngle: np.uint32, maximumAngle: np.uint32, angleSpacing: int = 1, threshold=1):
        # * Setup the program
        result = self.setup()
        self.imageBGRA = None
        # self.imageBGRA = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        self.minAngle = minimumAngle
        self.maxAngle = maximumAngle
        self.angleSpacing = angleSpacing
        self.threshold = threshold
        # self.nChannels = self.imageBGRA.shape[2]
        # self.padding = self.imageBGRA.strides[0] - self.width * \
        #     self.imageBGRA.strides[1] * self.imageBGRA.itemsize
        self.thetas = None
        self.diag_len = None
        self.rhos = None
        self.num_thetas = None
        self.accumulator = None

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

    def _define_ranges(self):
        #  Hough accumulator of θ
        #  vs ρ
        # . It is a 2D array with the number of rows equal to the number of ρ
        #  values and the number of columns equal to the number of θ
        #  values.
        self.thetas = np.deg2rad(
            np.arange(self.minAngle, self.maxAngle, self.angleSpacing))
        self.diag_len = np.ceil(
            np.sqrt(self.width * self.width + self.height * self.height))
        self.rhos = np.linspace(-self.diag_len/2,
                                self.diag_len/2, self.diag_len.astype(np.uint32))
        self.num_thetas = len(self.thetas)
        self.accumulator = np.zeros(
            (self.diag_len.astype(np.uint32), self.num_thetas), dtype=np.uint32)
        # print(self.diag_len)

    def _hough_plane(self):
        pass

    def _hough_lines(self):
        pass

    def _hough_lines_segments(self):
        pass

  

    def findLines(self):
        try:
            out_lines = []
            if self.accumulator is None:
                return
            np.sort(self.accumulator)[::-1]

            # for each collumn find the max rho? is that is
            # for thetaidx in range(len(255-self.accumulator)):
            for thetaidx in range(len(self.thetas)):

                rhoidx = np.argmax(self.accumulator[:, thetaidx])
                # print(self.rhos.shape)
                # print(self.thetas.shape)
                # rhoidx = np.argmax(self.accumulator[thetaidx][rhomax])
                # print(rhoidx)
                current_vote = rhoidx / self.num_thetas
                if current_vote >= self.threshold:
                    # a = math.cos(thetaidx)
                    a = math.cos(self.thetas[thetaidx])
                    b = math.sin(self.thetas[thetaidx])
                    # b = math.sin(thetaidx)
                    # x0 = a * thetaidx
                    scale = self.diag_len 
                    
                    x0 = a * self.rhos[int(rhoidx)] + self.width  /2
                    y0 = b * self.rhos[int(rhoidx)] + self.height /2
                    pt1 = (int(x0 + scale * (-b)), int(y0 + scale * (a)))
                    pt2 = (int(x0 - scale * (-b)), int(y0 - scale * (a)))
                    out_lines.append((pt1, pt2))

            # # print(out_lines)
            # # * remove duplicates
            # # for pos1, pos2 in out_lines:
            # #     if all(abs(pos1[0] - ))
            # line_sums = [np.sum(self.imageBGRA[min(pos1[1], pos2[1]):max(pos1[1], pos2[1]),
            #                                    min(pos1[0], pos2[0]):max(pos1[0], pos2[0])]) for pos1, pos2 in out_lines]

            # # * Draw the lines on the image
            # # Sort lines based on the sum of pixel values
            # sorted_lines = [line for _, line in sorted(
            #     zip(line_sums, out_lines), key=lambda x: x[0], reverse=True)]

            # # Initialize an empty mask to accumulate pixel values
            # pixel_accumulator = np.zeros_like(self.imageBGRA[:, :, 0], dtype=np.uint32)

            # # Keep only the top N lines
            # top_n_lines = []

            # for pos1, pos2 in sorted_lines:
            #     # Create a mask for the current line
            #     mask = np.zeros_like(self.imageBGRA[:, :, 0], dtype=np.uint8)
            #     cv.line(mask, pos1, pos2, 255, 1, cv.LINE_AA)

            #     # Accumulate pixel values in the mask
            #     pixel_accumulator += mask * self.imageBGRA[:, :, 0]

            #     top_n_lines.append((pos1, pos2))

            #     if len(top_n_lines) == 40:
            #         break

            #* Cluster lines
            # Compute line sums
            # line_sums = [np.sum(self.imageBGRA[min(pos1[1], pos2[1]):max(pos1[1], pos2[1]),
            #                                    min(pos1[0], pos2[0]):max(pos1[0], pos2[0])]) for pos1, pos2 in out_lines]

            # # Create a numpy array of lines for DBSCAN
            # lines_array = np.array(out_lines).reshape(-1, 4)

            # # Perform DBSCAN clustering
            # clustering = DBSCAN(eps=self.diag_len, min_samples=1)
            # labels = clustering.fit_predict(lines_array)

            # # Get unique labels (clusters)
            # unique_labels = np.unique(labels)

            # # Initialize an empty mask to accumulate pixel values
            # pixel_accumulator = np.zeros_like(
            #     self.imageBGRA[:, :, 0], dtype=np.uint32)

            # # Keep only the top N lines for each cluster
            # top_n_lines = []

            # for cluster_label in unique_labels:
            #     # Get lines belonging to the current cluster
            #     cluster_lines = lines_array[labels == cluster_label]

            #     # Sort lines based on the sum of pixel values
            #     sorted_cluster_lines = [line for _, line in sorted(
            #         zip(line_sums, cluster_lines), key=lambda x: x[0], reverse=True)]

            #     # Keep only the top N lines for each cluster
            #     top_n_lines.extend(sorted_cluster_lines[:10])

            # # * Draw the lines on the image
            # for pos1, pos2 in top_n_lines:
            #     cv.line(self.imageBGRA, (pos1[0], pos1[1]),
            #             (pos2[0], pos2[1]), (0, 0, 255), 1, cv.LINE_AA)

            # # * Draw the lines on the image
            # for pos1, pos2 in top_n_lines:
            #     cv.line(self.imageBGRA, pos1, pos2, (0, 0, 255), 1, cv.LINE_AA)


            # * Show the result
            # iF.showSideBySideImages(self.imageBGRA, 255-self.accumulator,"test", False,True)
            # cv.imshow("Hough", self.imageBGRA)
            # cv.waitKey(0)
            # return top_n_lines
            return out_lines
        except Exception as e:
            print(e)
            # TODO: Logger

    def GPUCalc(self, image: np.ndarray,):
        try:
            # * Convert Image to BGRA
            self.imageBGRA = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
            self.width = self.imageBGRA.shape[1]
            self.height = self.imageBGRA.shape[0]
            self.widthstep = self.imageBGRA.strides[0]
            self._define_ranges()
            imgFormat = cl.ImageFormat(
                cl.channel_order.BGRA,
                cl.channel_type.UNSIGNED_INT8
            )
            # * Accumulator Buffer
            accumulatorBuffer = cl.Buffer(
                ctx,
                flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_WRITE,
                size=self.accumulator.nbytes,
                hostbuf=self.accumulator
            )
            # * IMAGE Buffer In
            imageIn = cl.Image(
                ctx,
                flags=cl.mem_flags.COPY_HOST_PTR | cl.mem_flags.READ_ONLY,
                format=imgFormat,
                shape=(self.width, self.height),
                pitches=(self.imageBGRA.strides[0], self.imageBGRA.strides[1]),
                hostbuf=self.imageBGRA.data
            )

            # * Setup local and global working memory
            localws = (32, 16)  # openCV 32x8 = 256
            globalws = (math.ceil(self.width / localws[0]) * localws[0],
                        math.ceil(self.height / localws[1]) * localws[1])
            # * Send Paramenters to device
            kernelName = prog.vote_hough_accum
            kernelName.set_arg(0, imageIn)
            kernelName.set_arg(1, accumulatorBuffer)
            kernelName.set_arg(2, np.uint32(self.num_thetas))
            kernelName.set_arg(3, np.uint32(self.diag_len))
            kernelName.set_arg(4, np.uint32(self.minAngle))
            kernelName.set_arg(5, np.uint32(self.maxAngle))
            kernelName.set_arg(6, np.uint32(self.angleSpacing))
            # * Start the kernel program
            kernelEvent = cl.enqueue_nd_range_kernel(commQ,
                                                     kernelName,
                                                     global_work_size=globalws,
                                                     local_work_size=localws
                                                     ).wait()

            # * Get result form the program
            cl.enqueue_copy(
                commQ,
                dest=self.accumulator,
                src=accumulatorBuffer,
                is_blocking=True
            )
            foundLines = self.findLines()
            imageIn.release()
            accumulatorBuffer.release()
            return self.imageBGRA, foundLines 
        except Exception as e:
            # TODO: Attach logger
            print(e)
