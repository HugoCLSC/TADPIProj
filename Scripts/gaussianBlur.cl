// First Part Gaussian Filter (Gaussian Blur)

__constant sampler_t gaussSampler = CLK_NORMALIZED_COORDS_FALSE |
                                    CLK_ADDRESS_CLAMP_TO_EDGE |
                                    CLK_FILTER_NEAREST;

// __kernel void gaussian_blur(
//     __read_only image2d_t imgIn,
//     __constant float * mask,
//     __global float * blurr,
//     __private int maskSize
// ){
//     const int2 pos = {get_global_id(0), get_global_id(1)};

//     float sum = 0.0f;

//     for(int a = -maskSize; a < maskSize+1; a++){
//         for(int b = -maskSize; b < maskSize+1; b++){
//             sum += mask[a+ maskSize + (b + maskSize) * (maskSize*2 + 1)]
//                 *read_imagef(imgIn, gaussSampler, pos + (int2)(a,b)).x;
//         }
//     }

//     blurr[pos.x+pos.y*get_global_size(0)] = sum;
// }


__kernel void gauss(
    __read_only image2d_t imageIn,
    __write_only image2d_t imageOut,
    __constant float *mask,
    const int kSize,
    sampler_t gaussSampler
){
    //Coordinates 
    int iX = get_global_id(0);
    int iY = get_global_id(1);
    const int center = kSize / 2;
    const int2 gid = (int2)(get_global_id(0), get_global_id(1));
    float sum = 0.0f;
    
    for(int i = 0; i < kSize; ++i){
        for(int j = 0; j < kSize; ++j){
            int2 offset = (int2)(i - center, j - center);
            int2 coord = gid + offset;
            float pixel = read_imagef(imageIn, gaussSampler, coord).x;
            sum += mask[i] * mask[j] * pixel;
        }
    }

    write_imagef(imageOut, gid, (float4)(sum, 0.0f, 0.0f, 0.0f));

}