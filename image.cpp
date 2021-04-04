#include <iostream>

#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;


void slidingWindowCPU(cv::Mat img, int h, int w, int stride){
    for(int row=0; row <= img.rows; row+=stride){
        for (int col=0; col <= img.cols; col+=stride){
            cv::Rect windows(col, row, h, w);
            //cv::Mat Roi = img(windows);

            cv::rectangle(img, windows, cv::Scalar(255), 1, 8, 0);
            // cv::imshow("Image", img);
            // cv::waitKey(1);
        }
    }
}


int main(){

    std::string impath {"/home/mvinceslas/coding/cpp_cuda/nvidia_accelerated_computing/sat.png"};
    cv::Mat img = cv::imread(impath, cv::IMREAD_COLOR);

    cout<< img.cols << " " << img.rows <<endl;

    //cv::Point pt1 = cv::Point(100, 300);
    //cv::circle(img, pt1, 3, cv::Scalar(0, 0, 255), -1, cv::LINE_8);


    int h = 10;
    int w = 10;
    int stride = 10;

    slidingWindowCPU(img, h, w, stride);

    // cv::namedWindow("Image", cv::WINDOW_AUTOSIZE);
    // cv::imshow("Image", img);
    // cv::waitKey(0);

    cv::imwrite("test.png", img);
    cv::destroyAllWindows();

    std::cout<<"\n\n";
    return 0;
}