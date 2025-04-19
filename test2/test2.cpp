#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main (){
    cv::Mat image(400, 600, CV_8UC3, cv::Scalar(255, 0, 0)); //blue image
    cv::imshow("blue window", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}