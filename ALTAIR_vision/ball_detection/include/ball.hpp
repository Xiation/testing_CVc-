#ifndef BALL
#define BALL

#include <opencv2/opencv.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <numeric>


// Global variables to mimic the Python global variables (center_x, center_y)
extern int center_x ;
extern int center_y ;

// Function declarations
cv::Mat ball(const cv::Mat &frame) ;
std::pair<cv::Mat, cv::Mat> field(const cv::Mat &frame);
std::pair<int, int> find_first_last_orange(const cv::Mat &line_data, int line_start) ;
std::pair<int, int> find_top_bottom_orange(const cv::Mat &column_data, int col_start) ;
std::vector<double> movingAverage(const std::vector<double>& data, int windowSize) ;
cv::Mat detect(const cv::Mat &mask_ball, const cv::Mat &mask_field, cv::Mat frame, std::vector<double>& distanceHistory, int movingAverageWindow) ; 



#endif