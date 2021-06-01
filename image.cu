#include <iostream>
#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;
using cv::Size;


void slidingWindowGPU(cv::Mat img, int h, int w, int stride){
    for(int row=0; row <= img.rows; row+=stride){
        for (int col=0; col <= img.cols; col+=stride){
            cv::Rect windows(col, row, h, w);
            cv::rectangle(img, windows, cv::Scalar(row, col, 0), 1, 8, 0);
        }
    }
}


int main(){

    std::string impath {"/home/mdhvince/coding/accelerated_computing/sat.png"};
    cv::Mat img = cv::imread(impath, cv::IMREAD_COLOR);
    cv::resize(img, img, Size(512, 512));
    cout<< img.cols << " " << img.rows <<endl;

    // size_t bytes = img.total() * sizeof(float);
    // cv::Mat d_img;
    // cudaMalloc(d_img, bytes);

    int h = 10;
    int w = 10;
    int stride = 10;

    slidingWindowGPU(img, h, w, stride);

    //cv::imwrite("/home/mdhvince/coding/accelerated_computing/testGPU.png", img);
    cv::destroyAllWindows();

    // cudaFree(d_img);

    return 0;
}