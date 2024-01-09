

__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | // Natural coorfinates
    CLK_ADDRESS_CLAMP_TO_EDGE |   // Clamp to zeros
    CLK_FILTER_NEAREST;

kernel void hough_lines(read_only image2d_t imgIn, __global int *accumulator,
                        int w, int h, int num_thetas) {}