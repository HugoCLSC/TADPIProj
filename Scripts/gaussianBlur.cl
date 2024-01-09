__constant sampler_t sampler_gauss =
    CLK_NORMALIZED_COORDS_FALSE | // Natura coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE |   // Clamp to zeros
    CLK_FILTER_LINEAR;

__kernel void gaussianBLUR_image2D(__read_only image2d_t imageIn,
                                   __write_only image2d_t imageOut, const kSize,
                                   __constant float *mask, int w, int h) {
  const int center = kSize / 2;
  // printf("HJELLE");
  // printf("%f", mask);
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  // printf("Work Item %d\n", gid);

  __local float4 sum;
  __local float sumGray;
  if (gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h) {

    for (int i = 0; i < kSize; ++i) {
      for (int j = 0; j < kSize; ++j) {
        int2 offset = (int2)(i - center, j - center);
        int2 coord = gid + offset;
        float4 pixel = read_imagef(imageIn, sampler_gauss, coord);
        double Gray = 0.21 * pixel.x + 0.72 * pixel.y + 0.07 * pixel.z;
        //* For Rgb
        // sum.x += mask[i] * mask[j] * pixel.x;
        // sum.y += mask[i] * mask[j] * pixel.y;
        // sum.z += mask[i] * mask[j] * pixel.z;
        sumGray += mask[i] * mask[j] * Gray;
      }
    }
    // Print the intermediate result using printf
    // printf("Work Item (%d, %d), Sum: %f\n", gid.x, gid.y, sum);

    // write_imagef(imageOut, gid, (float4)(sum , 0.0f, 0.0f, 0.0f));
    write_imagef(imageOut, gid, (float4)(sumGray, sumGray, sumGray, 0.0f));
    // write_imagef(imageOut, gid, (float4)(sum));
  }
}