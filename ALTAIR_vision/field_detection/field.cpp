#include <opencv2/opencv.hpp>
#include <iostream> 
#include <vector>

// temporary color table for preprocess
cv::Scalar lowerGreen(35,100,100);
cv::Scalar upperGreen(85,255,255);

cv::Mat preprocess(const cv::Mat &frame){
    if (frame.empty()){
        throw std:: invalid_argument("Invalid frame or frame is empty");
     }
    cv::Mat hsv, greenMask, greenMaskCleaned;

    //  change image format from rgb to hsv
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV) ;

    //  field green masking
    cv::inRange(hsv, lowerGreen, upperGreen, greenMask);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3));

    cv::morphologyEx(greenMask, greenMaskCleaned, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 3);

    std::vector<std::vector<cv::Point>>countours;
    cv::findContours(greenMaskCleaned, countours, cv::RETR_EXTERNAL, cv:: CHAIN_APPROX_SIMPLE);
    
    // find largest countours
    std::vector<cv::Point> largest_countour ;
    if (!countours.empty()) {
        largest_countour = *std::max_element(countours.begin(), countours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) 
        {
        return cv::contourArea(a) < cv::contourArea(b);
        });
    }
    // create a mask for ROI
    cv::Mat roiMask = cv::Mat::zeros(greenMaskCleaned.size(), CV_8U);

    if (!largest_countour.empty()) {
        std::vector<cv::Point> hull;
        cv::convexHull(largest_countour, hull) ;
        cv::drawContours(roiMask, std::vector<std::vector<cv::Point>>{hull}, -1, 255, -1) ;
    }

    // create ROI image using bitwise AND
    cv::Mat roi_image;
    if(!largest_countour.empty()) {
        cv::bitwise_and(frame, frame, roi_image, roiMask) ;
        return roi_image;
    }
    return frame ;
}

int main (){
    cv::Mat frame = cv::imread("/home/abyan/Documents/testing_CVc++/ALTAIR_vision/samplesIMG/sampleLineDet.jpeg") ;
    
    try{
        cv::Mat result = preprocess(frame);
        cv::imshow("Processed image" , result);

        while(true){
            char key = cv::waitKey(0);
            if (key == 'q') {
                std::cout << "exiting program" << std::endl;
                break;
            } else {
                std::cout << "invalid terminate key. You press: " << char(key) << std::endl ;
            }
        }
        cv::destroyAllWindows();
    } catch (const std::invalid_argument &e){
        std::cerr << e.what() << std::endl;
    }
}