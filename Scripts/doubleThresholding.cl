__constant sampler_t sampler_doublet =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void double_threshold_image2D(__read_only image2d_t imageIn,
                                       __write_only image2d_t imageOut,
                                       int padding, float lowThresholdRatio,
                                       float highThresholdRatio, int w, int h,
                                       float maxValue) {
  int2 dim = get_image_dim(imageIn);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));

  uint4 weakValue = (25, 25, 25);
  uint4 strongValue = (255, 255, 255);
  uint4 surpressedValue = (0, 0, 0);
  if (gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h) {
    double highThreshold = maxValue * highThresholdRatio;
    double lowThreshold = highThreshold * lowThresholdRatio;
    // printf("max %f\n", highThreshold);
    // printf("min %f\n", lowThreshold);
    uint4 pixel = read_imageui(imageIn, sampler_doublet, gid);
    // printf("Pixel value: %d \n", pixel.x);

    float4 pixelf = convert_float4_rte(pixel);

    if (pixelf.x > highThreshold) {
      write_imageui(imageOut, (int2)gid, (uint4)(strongValue));
    } else if (pixelf.x <= lowThreshold) {
      write_imageui(imageOut, (int2)gid, (uint4)(surpressedValue));
    } else if (pixelf.x <= highThreshold && pixelf.x >= lowThreshold) {
      write_imageui(imageOut, (int2)gid, (uint4)(weakValue));
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}
__constant sampler_t sampler_hysteresis =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void hysteresis_image2D(__read_only image2d_t imageIn,
                                 __write_only image2d_t imageOut, int w,
                                 int h) {
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  uint4 weakValue = (25, 25, 25);
  uint4 surpressedValue = (0, 0, 0);
  uint4 strongValue = (255, 255, 255);
  __local int strong;
  //   barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

  if (gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h) {
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);

    uint4 Pixel11 =
        read_imageui(imageIn, sampler_hysteresis, (int2)(gid.x, gid.y));
    if (Pixel11.x == weakValue.x) {

      uint4 Pixel00 = read_imageui(imageIn, sampler_hysteresis,
                                   (int2)(gid.x - 1, gid.y - 1));
      uint4 Pixel01 =
          read_imageui(imageIn, sampler_hysteresis, (int2)(gid.x, gid.y - 1));
      uint4 Pixel02 = read_imageui(imageIn, sampler_hysteresis,
                                   (int2)(gid.x + 1, gid.y - 1));

      uint4 Pixel10 =
          read_imageui(imageIn, sampler_hysteresis, (int2)(gid.x - 1, gid.y));
      uint4 Pixel12 =
          read_imageui(imageIn, sampler_hysteresis, (int2)(gid.x + 1, gid.y));

      uint4 Pixel20 = read_imageui(imageIn, sampler_hysteresis,
                                   (int2)(gid.x - 1, gid.y + 1));
      uint4 Pixel21 =
          read_imageui(imageIn, sampler_hysteresis, (int2)(gid.x, gid.y + 1));
      uint4 Pixel22 = read_imageui(imageIn, sampler_hysteresis,
                                   (int2)(gid.x + 1, gid.y + 1));
      if ((Pixel00.x == strongValue.x || Pixel01.x == strongValue.x ||
           Pixel02.x == strongValue.x || Pixel10.x == strongValue.x ||
           Pixel12.x == strongValue.x || Pixel20.x == strongValue.x ||
           Pixel21.x == strongValue.x || Pixel22.x == strongValue.x)) {
        write_imageui(imageOut, (int2)gid, (uint4)(strongValue));
      } else {
        write_imageui(imageOut, (int2)gid, (uint4)(surpressedValue));
      }
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}