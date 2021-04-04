#include <iostream>

#include <opencv2/opencv.hpp>

using std::cout;
using std::endl;


int main()
{
    cv::Mat frame;
    cv::VideoCapture cap;

    int deviceID = 0;
    int apiID = cv::CAP_ANY;

    cap.open(deviceID, apiID);
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }

    while (cap.isOpened()) {
        cap.read(frame);
    
        if (frame.empty())
            break;



        
        cv::imshow("Live", frame);
        if (cv::waitKey(5) >= 0)
            break;
    }

    return 0;
}