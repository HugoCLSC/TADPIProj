__constant sampler_t sampler_non_max =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void non_max_suppression_image2D(__read_only image2d_t imageIn,
                                          __write_only image2d_t imageOut,
                                          int padding,
                                          __global double *angleMatrix,
                                          const int w, const int h) {

  int2 dim = get_image_dim(imageIn);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  __local float4 pixelf;
  __local double angle;
  uint4 gg = (0, 0, 0, 1.0f);

  if (gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h) {
    //* From radians to degrees
    angle = ((angleMatrix[gid.x + gid.y * w] * 180.0f) / M_PI);
    if (angle < (double)0.0f) {
      angle += 180.0f;
      //   printf("angle bellow. %f\n", angle);
    }
    // barrier(CLK_LOCAL_MEM_FENCE);
    // angle = (double)fmod(((angleMatrix[gid.x + gid.y * w]* 180.0f) / M_PI) +
    // 360.0f,360);
    if (angle == NAN) {
      printf("No value");
    }
    // se e menor que zero adiciona 180
    uint q = 255.0f;
    uint r = 255.0f;
    // * Angle 0
    if ((angle >= 0.0f && angle < 22.5f) ||
        (angle >= 157.5f && angle <= 180.0f)) {
      q = read_imageui(imageIn, sampler_non_max, (int2)(gid.x, gid.y + 1)).x;
      r = read_imageui(imageIn, sampler_non_max, (int2)(gid.x, gid.y - 1)).x;
    }
    // * Angle 25
    else if (angle >= 22.5f && angle < 67.5f) {
      q = read_imageui(imageIn, sampler_non_max, (int2)(gid.x + 1, gid.y - 1))
              .x;
      r = read_imageui(imageIn, sampler_non_max, (int2)(gid.x - 1, gid.y + 1))
              .x;
    }
    // * Angle 90
    else if (angle >= 67.5f && angle < 112.5f) {
      q = read_imageui(imageIn, sampler_non_max, (int2)(gid.x + 1, gid.y)).x;
      r = read_imageui(imageIn, sampler_non_max, (int2)(gid.x - 1, gid.y)).x;
    }
    // * Angle 135
    else if (angle >= 112.5f && angle < 157.5f) {
      q = read_imageui(imageIn, sampler_non_max, (int2)(gid.x - 1, gid.y - 1))
              .x;
      r = read_imageui(imageIn, sampler_non_max, (int2)(gid.x + 1, gid.y + 1))
              .x;
    }
    uint pixel = read_imageui(imageIn, sampler_non_max, gid).x;
    pixelf = convert_float4_rte(pixel);
    if (pixel >= q && pixel >= r) {
      write_imageui(imageOut, (int2)gid, (uint4)(pixel, pixel, pixel, 1.0f));
    } else {
      write_imageui(imageOut, (int2)gid, (uint4)(gg));
    }
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}