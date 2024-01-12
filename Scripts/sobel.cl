__constant sampler_t sampler_sobel =
    CLK_NORMALIZED_COORDS_FALSE | // Natura coordinates
    CLK_ADDRESS_CLAMP_TO_EDGE |   // Clamp to zeros
    CLK_FILTER_NEAREST;

__kernel void sobel_image2D(__read_only image2d_t imageIn,
                            __write_only image2d_t imageOut,
                            int t1, int t2, int w, int h) {

    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    if(gid.x >= 0 && gid.x < w && gid.y >= 0 && gid.y < h){

        uint4 Pixel00 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x - 1, gid.y - 1));
        uint4 Pixel01 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x, gid.y - 1));
        uint4 Pixel02 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x + 1, gid.y - 1));

        uint4 Pixel10 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x - 1, gid.y));
        uint4 Pixel12 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x + 1, gid.y));

        uint4 Pixel20 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x - 1, gid.y + 1));
        uint4 Pixel21 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x, gid.y + 1));
        uint4 Pixel22 = read_imageui(imageIn, sampler_sobel, (int2)(gid.x + 1, gid.y + 1));

        uint4 Gx = Pixel00 + (2 * Pixel10) + Pixel20 - Pixel02 - (2 * Pixel12) - Pixel22;
        uint4 Gy = Pixel00 + (2 * Pixel01) + Pixel02 - Pixel20 - (2 * Pixel21) - Pixel22;

        uint4 G = (uint4)(0, 0, 0, Pixel00.w);
        G.x = sqrt((float)(Gx.x * Gx.x + Gy.x * Gy.x)); // B
        G.y = sqrt((float)(Gx.y * Gx.y + Gy.y * Gy.y)); // G
        G.z = sqrt((float)(Gx.z * Gx.z + Gy.z * Gy.z)); // R

        double diff = abs((int)(G.z-G.y)) + abs((int)(G.z-G.x)) + abs((int)(G.y-G.x));
        double average = (G.x+G.y+G.z)/3;

        if(diff>t1 && average>t2){
        write_imageui( imageOut, (int2)(gid.x,gid.y) , (uint4)(255,255,255,0));
        }
        else
            write_imageui( imageOut, (int2)(gid.x,gid.y) , (uint4)(0,0,0,0));

    }

}

// Realce de contornos operatores de sobel
// Sx = (a + 2d +g ) - (c + 2f + i)
// Sy = (g + 2h + i) - (a + 3b + c)
// S = sqrt(Sx^2 + Sy^2)
// S = Abs(Sx) + Abs(Sy)