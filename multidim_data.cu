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
void scalePixel(float *d_inp_img, float *d_out_img, int height, int width){
    row_threadId_in_grid = blockIdx.y * blockDim.y + threadIdx.y;               // thread id on y within the grid
    col_threadId_in_grid = blockIdx.x * blockDim.x + threadIdx.x;               // thread id on x within the grid

    if((row_threadId_in_grid < height) && (col_threadId_in_grid < width)){
        int pixel_loc = row_threadId_in_grid * width + col_threadId_in_grid;     // also the thread location linearized 1D
        d_out_img[pixel_loc] = 2.0 * d_inp_img[pixel_loc];
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