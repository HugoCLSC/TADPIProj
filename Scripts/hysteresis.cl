__constant sampler_t sampler_hysteresis=
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

__kernel void hysteresis_image2D(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    int padding, 

)