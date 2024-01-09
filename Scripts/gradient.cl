__constant sampler_t sampler_gradient =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
// https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/
// https://en.wikipedia.org/wiki/Kernel_(image_processing)
// https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

__kernel void gradient_image2D(__read_only image2d_t imageIn,
                               __write_only image2d_t imageOut, const int w,
                               const int h, __global double *angleMatrix) {
  //* Sobel
  // float Kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  // float Ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  //* Scharrr
  float Kx[3][3] = {{+3, 0, -3}, {+10, 0, -10}, {+3, 0, -3}};
  float Ky[3][3] = {{+3, +10, +3}, {0, 0, 0}, {-3, -10, -3}};
  //* Width and height of Kx and Ky
  int2 KParameters = (int2)(3, 3);
  int2 center = (int2)(1, 1);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  // __local float4 sumX;
  // __local float4 sumY;
  __local double sumX;
  __local double sumY;
  __local double grad;
  __local uint4 gg;
  int i = get_global_id(0) + get_global_id(1) * get_global_size(0);
  int idx = i + (i / w) * 9 * 4;
  // * don't know which memory i have affected so jsut do both i guess
  if (gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h) {
    for (int i = 0; i < KParameters.x; ++i) {
      for (int j = 0; j < KParameters.y; ++j) {
        int2 offset = (int2)(i - center.x, j - center.y);
        int2 coord = gid + offset;
        uint pixel = read_imageui(imageIn, sampler_gradient, coord).x;
        double pixelf = convert_double_rte(pixel);
        sumX += Kx[i][j] * pixelf;
        sumY += Ky[i][j] * pixelf;
      }
    }
    // double nomSumX = sqrt(sumX * sumX);
    // double nomSumY = sqrt(sumY * sumY);
    grad = sqrt((sumX * sumX) + (sumY * sumY)); // THIS ONE LOOKS GOOD
    gg = convert_uint4_rte((double4)(grad, grad, grad, 0.0f));
    write_imageui(imageOut, (int2)gid, (uint4)(gg));
    // No need to normalize directions of the gradient atan is -pi to pi
    double theta = atan2(sumY, sumX);
    angleMatrix[gid.x + gid.y * w] = theta;
    // printf("Angle : %f\n", theta);
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}

// abs(G) = sqrt(Ix^2 + Iy^2)
// ro(x,y) = arctang(Ix/Iy)iiiiiiiiiiiiiiiiiiiiii
