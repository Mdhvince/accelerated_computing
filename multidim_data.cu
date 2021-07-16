/* 
Active learning with the book: programming massively parallel processors
@medhyvinceslas
*/

#include <iostream>


/* 
- all cuda thread within a grid execute the same kernel
- they are distinguishable by their coordinates
- max dim_block (nb thread) in one block is 1024, distributed accross all dimension
- Allowed values of gridDim.x .y .z (nb blocks on each direction) range from 1 to 65,536 in CUDA
*/

/* linearize 2D data index
    - Matrix M of 3 rows, 4 column
    - 2D index of M(y, x) is : y, x
    - Transforming into 1D index M(y, x) is : y * 3 + x   
    --> multiply row_index by number of element in each row (width) and add the col_index
*/

__global__
void scalePixel(unsigned char *d_inp_img, unsigned char *d_out_img, const int height, const int width){
    // where is the current thread on the grid
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < height) && (col < width)){
        // At this moment, if the grid would match perfectly the data, the thread
        // coordinates (row, col) would be the pixel coordinates too, and we could
        // access a particular pixel with img[row][col]. But the grid is in general
        // larger than the data. So we need to linearize the pixel location in 1D

        const int pixel_loc = row * width + col;
        d_out_img[pixel_loc] = 2.0 * d_inp_img[pixel_loc];
    }
}


__global__
void rgbToGray(unsigned char *rgbImg, unsigned char *grayImg, const int height, const int width){
    /*
    In an rgb image, pixels are represented as a linear combinations of r, g, b channels.
    So each pixels will be represented as (1-y-x)*r + x*g + y*b = pixel intensity.
    if we have and image 3 pixels wide, it will be represented as : r0, g0, b0, r1, g1, b1, r2, g2, b2.
    So every 3 iteration represent one pixel.

    A grayscale image is an image where the value of the pixels carries only intensity information.
    To obtain grayscale img from RGB, we use the formula : 0.21*r + 0.71*g + 0.07*b
    */
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if((row < height) && (col < width)){
        // gray converted pixel will be storage at this location (1D coordinate)
        const int gray_pixel_loc = row * width + col;

        // pixel gray at 0, map to pixel rgb starting at 0 (r0).
        // pixel gray at 1, map to pixel rgb starting at 3 (r1).
        // pixel gray at 2, map to pixel rgb starting at 6 (r2).
        // so rgb_pixel_loc map to 3 * gray_pixel_loc

        const int rgb_pixel_loc = 3 * gray_pixel_loc;
        unsigned char r = rgbImg[rgb_pixel_loc    ];
        unsigned char g = rgbImg[rgb_pixel_loc + 1];                   // the next pixel on rx, gx, bx
        unsigned char b = rgbImg[rgb_pixel_loc + 2];

        grayImg[gray_pixel_loc] = .21f*r + .71f*g + .07f*b;

    }
}

__global__
void blurKernel(unsigned char *inp_img, unsigned char *out_img, const int height, const int width, const int blur_size){
    /*Img blurring
    - for each tread, take the output pixel position it is mapped to
    - draw a surrounding box around this pixel
    - take all the pixels inside this boxe
    - average them
    - blurred pixel value = the average
    */

    // central pixel location of the patch
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    // blur_size = 2; so 2pixels on the left of the center pixel, 2 on the right, up and down
    // so the total patch size is 2(left)+2(right)+1(center) = 5pixels

    if((row < height) && (col < width)){
        int pixels_sum {0};
        int nb_pixels_added {0};

        for(int row_patch=-blur_size; row_patch<blur_size+1; ++row_patch){          // iterate vertically through the patch
            for (int col_patch=-blur_size; col_patch<blur_size+1; ++col_patch){         // iterate horizontally through the patch
                
                const int current_row = row + row_patch;                            // current row in the grid
                const int current_col = col + col_patch;

                if(current_row > -1 && current_row < height && current_col > -1 && current_col < width){   // we are in the image range
                    pixels_sum += inp_img[current_row * width + current_col];
                    ++nb_pixels_added;
                }
            }
        } // end going through one patch

        out_img[row * width + col] = static_cast<unsigned char>(pixels_sum / nb_pixels_added);     // average the patch (Blur)

    } 

    
}



int main(){

    int data_dim_x {78};
    int data_dim_y {64};

    dim3 dim_block (16, 16, 1);                                       // 16x16x1 = 256 threads per block
    size_t dim_grid_x = (data_dim_x + 16 - 1) / 16;                   // ensure nb_block covers the data on x and y
    size_t dim_grid_y = (data_dim_y + 16 - 1) / 16;                   // we can also use ceil(data_dim_y / 16.0), float is important
    dim3 dim_grid (dim_grid_x, dim_grid_y, 1);



    std::cout<< ceil(78 / 16.)  <<std::endl;

    return 0;
}