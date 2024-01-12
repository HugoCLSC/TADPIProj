__constant sampler_t sampler_gauss =
    CLK_NORMALIZED_COORDS_FALSE | // Natura coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE |   // Clamp to zeros
    CLK_FILTER_LINEAR;

__kernel void gaussianBLUR_image2D(__read_only image2d_t imageIn,
                                   __write_only image2d_t imageOut, const kSize,
                                   __constant float *mask, int w, int h) {
  const int center = kSize / 2;
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  __local uint4 sum;
  __local uint4 s;
  if (gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h) {

    for (int i = 0; i < kSize; ++i) {
      for (int j = 0; j < kSize; ++j) {
        int2 offset = (int2)(i - center, j - center);
        int2 coord = gid + offset;
        uint4 pixel = read_imageui(imageIn, sampler_gauss, coord);
        //* For Rgb
        sum.x += mask[i] * mask[j] * pixel.x;
        sum.y += mask[i] * mask[j] * pixel.y;
        sum.z += mask[i] * mask[j] * pixel.z;
      }
    }
    // Print the intermediate result using printf
    // printf("Work Item (%d, %d), Sum: %f\n", gid.x, gid.y, sum.x);
    write_imageui(imageOut, gid, (uint4)(sum));
    barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);
  }
}

// @ Make it Gray ---> Gray = 0.21 R + 0.72 G + 0.07 B