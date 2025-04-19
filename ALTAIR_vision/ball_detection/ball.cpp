#include "ball.hpp"

// Global variable definitions
int center_x = 0;
int center_y = 0;

cv::Mat ball(const cv::Mat &frame) {
    // Convert to HSV color space
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // Define lower and upper bounds for orange color
    cv::Scalar lower_orange(1, 80, 20);
    cv::Scalar upper_orange(25, 255, 255);

    // Threshold the HSV image to get only orange colors
    cv::Mat mask_ball;
    cv::inRange(hsv, lower_orange, upper_orange, mask_ball);

    // Create a kernel of ones for morphological operations
    cv::Mat kernel = cv::Mat::ones(15, 15, CV_8U);
    cv::morphologyEx(mask_ball, mask_ball, cv::MORPH_CLOSE, kernel);

    return mask_ball;
}

std::pair<cv::Mat, cv::Mat> field(const cv::Mat &frame) {
    // Define lower and upper bounds for green color
    cv::Scalar low_green(30, 30, 45);
    cv::Scalar up_green(85, 255, 255);

    // Convert the frame to HSV color space
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
    
    // Create an elliptical structuring element (kernel) for morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    
    // Threshold the HSV image to get only green colors
    cv::Mat green_mask;
    cv::inRange(hsv, low_green, up_green, green_mask);
    
    // Perform erosion followed by dilation
    cv::Mat green_mask1, green_mask2;
    cv::erode(green_mask, green_mask1, kernel, cv::Point(-1,-1), 1);
    cv::dilate(green_mask1, green_mask2, kernel, cv::Point(-1,-1), 6);

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(green_mask2, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        // If no contours found, return the original frame and an empty mask of the same size as green_mask2
        return std::make_pair(frame, cv::Mat::zeros(green_mask2.size(), green_mask2.type()));
    }
    
    // Find the largest contour based on area
    double max_area = 0;
    int max_index = -1;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_index = static_cast<int>(i);
        }
    }
    
    std::vector<cv::Point> largest_contour = contours[max_index];
    
    // Compute the convex hull of the largest contour
    std::vector<cv::Point> hull;
    cv::convexHull(largest_contour, hull);

    // Create a mask of zeros and draw the convex hull on it
    cv::Mat mask = cv::Mat::zeros(green_mask2.size(), green_mask2.type());
    std::vector<std::vector<cv::Point>> hulls;
    hulls.push_back(hull);
    cv::drawContours(mask, hulls, -1, cv::Scalar(255), cv::FILLED);
    
    // Create ROI by masking the original frame
    cv::Mat roi_frame;
    cv::bitwise_and(frame, frame, roi_frame, mask);
    
    return std::make_pair(roi_frame, mask);
}

std::pair<int, int> find_first_last_orange(const cv::Mat &line_data, int line_start) {
    int first_orange = -1;
    int last_orange = -1;
    
    // Loop from beginning of the line_data row
    for (int idx = 0; idx < line_data.cols; idx++) {
        if (line_data.at<uchar>(0, idx) == 255) {
            first_orange = line_start + idx;
            break;
        }
    }
    
    // Loop from end of the line_data row backward
    for (int idx = line_data.cols - 1; idx >= 0; idx--) {
        if (line_data.at<uchar>(0, idx) == 255) {
            last_orange = line_start + idx;
            break;
        }
    }
    
    return std::make_pair(first_orange, last_orange);
}

std::pair<int, int> find_top_bottom_orange(const cv::Mat &column_data, int col_start) {
    int top_orange = -1;
    int bot_orange = -1;
    
    // Loop from top of the column_data
    for (int idy = 0; idy < column_data.rows; idy++) {
        if (column_data.at<uchar>(idy, 0) == 255) {
            top_orange = col_start + idy;
            break;
        }
    }
    
    // Loop from bottom of the column_data upward
    for (int idy = column_data.rows - 1; idy >= 0; idy--) {
        if (column_data.at<uchar>(idy, 0) == 255) {
            bot_orange = col_start + idy;
            break;
        }
    }
    
    return std::make_pair(top_orange, bot_orange);
}

// function for moving average
std::vector<double> movingAverage(const std::vector<double>& data, int windowSize) {
    std::vector<double> smoothedData;
    if (data.empty() || windowSize <- 0) {
        return smoothedData; // return empty for invalid input
    }
    for (int i = 0; i < data.size(); ++i) {
        int start = std::max(0, static_cast<int>(i) - windowSize + 1); // ensure start isn't negative

        // extract window of data
        std::vector<double> window;
        for (int j = start; j <= i; ++j){
            window.push_back(data[j]);
        }
        // calculate the average of the window
        double sum = std::accumulate(window.begin(), window.end(), 0.0);
        smoothedData.push_back(sum / window.size());
    }
    return smoothedData;
}

cv::Mat detect(const cv::Mat &mask_ball, const cv::Mat &mask_field, cv::Mat frame, std::vector<double>& distanceHistory, int movingAverageWindow) {
    
    cv::Mat blurred_ball_mask;
    cv::GaussianBlur(mask_ball, blurred_ball_mask, cv::Size(9, 9), 2);

    // Detect circles using HoughCircles
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(blurred_ball_mask, circles, cv::HOUGH_GRADIENT, 1.2, 50, 50, 30, 1, 1000);

    if (!circles.empty()) {
        // Convert circle parameters to integer values (the rounding is achieved by casting after rounding)
        for (size_t i = 0; i < circles.size(); i++) {
            int x = cvRound(circles[i][0]);
            int y = cvRound(circles[i][1]);
            int r = cvRound(circles[i][2]);
            int stroke = static_cast<int>(1.1 * r);
            
            // Horizontal line processing
            int line_y = y;
            int line_x_start = std::max(0, x - stroke);
            int line_x_end = std::min(mask_ball.cols - 1, x + stroke);
            // Get a horizontal slice (row) from mask_ball with proper range
            cv::Rect hRect(line_x_start, line_y, line_x_end - line_x_start, 1);
            if (hRect.width <= 0) {
                )tinue;
            }
            cv::Mat orange_hline = mask_ball(hRect).clone();
            // Reshape to 1 row for processing
            orange_hline = orange_hline.reshape(0, 1);
            
            std::pair<int, int> first_last = find_first_last_orange(orange_hline, line_x_start);
            int first_orange = first_last.first;
            int last_orange = first_last.second;
            
            if (first_orange == -1 || last_orange == -1) {
                continue;
            }
            
            int total_x_pixel = last_orange - first_orange;
            int r_new = total_x_pixel / 2;
            int x_new = first_orange + r_new;
            
            // Vertical line processing
            int line_x = x_new;
            int line_y_start = y - stroke;
            int line_y_end = y + stroke;

            // Clamp vertical boundaries
            line_y_start = std::max(0, line_y_start);
            line_y_end = std::min(mask_ball.rows - 1, line_y_end);
            if (line_y_end - line_y_start <= 0) {
                continue;
            }
            cv::Rect vRect(line_x, line_y_start, 1, line_y_end - line_y_start);
            cv::Mat orange_vline = mask_ball(vRect).clone();

            // Reshape to single column
            orange_vline = orange_vline.reshape(0, orange_vline.rows);
            
            std::pair<int, int> top_bot = find_top_bottom_orange(orange_vline, line_y_start);
            int top_orange = top_bot.first;
            int bot_orange = top_bot.second;
            
            if (top_orange == -1 || bot_orange == -1) {
                continue;
            }
            
            int total_y_pixel = abs(top_orange - bot_orange);
            int y_new = bot_orange - (total_y_pixel / 2);
            
            if (y_new != y) {
                line_y = y_new;
                cv::Rect hRect2(line_x_start, line_y, line_x_end - line_x_start, 1);
                if (hRect2.width <= 0) {
                    continue;
                }
                cv::Mat orange_hline2 = mask_ball(hRect2).clone();
                orange_hline2 = orange_hline2.reshape(0, 1);
                std::pair<int, int> first_last2 = find_first_last_orange(orange_hline2, line_x_start);
                first_orange = first_last2.first;
                last_orange = first_last2.second;
                
                if (first_orange == -1 || last_orange == -1) {
                    continue;
                }
                
                total_x_pixel = last_orange - first_orange;
                r_new = total_x_pixel / 2;
                x_new = first_orange + r_new;
            }
            
            int R = static_cast<int>(r_new * 1.5);  // jarak deteksi
            
            int x1 = std::max(x_new - R, 0);
            int y1 = std::max(y_new - R, 0);
            int x2 = std::min(x_new + R, frame.cols);
            int y2 = std::min(y_new + R, frame.rows);
            // cv::rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);

            // Compute the ratio of green in the surrounding field area
            cv::Rect surroundingRect(x1, y1, x2 - x1, y2 - y1);
            cv::Mat surrounding_field = mask_field(surroundingRect);
            double field_pixels = cv::countNonZero(surrounding_field);
            double field_ratio = field_pixels / (surrounding_field.size().area());

            // Compute the ratio of orange in the surrounding ball area
            cv::Mat surrounding_ball = mask_ball(surroundingRect);
            double ball_pixels = cv::countNonZero(surrounding_ball);
            double ball_ratio = ball_pixels / (surrounding_ball.size().area());

            if ((field_ratio > 0.16) && (ball_ratio < 0.47)) {
                cv::line(frame, cv::Point(x_new, y_new + r_new), cv::Point(x_new, y_new - r_new), cv::Scalar(0, 255, 0), 2);
                cv::line(frame, cv::Point(x_new - r_new, y_new), cv::Point(x_new + r_new, y_new), cv::Scalar(0, 255, 0), 2);
                cv::circle(frame, cv::Point(x_new, y_new), r_new, cv::Scalar(0, 255, 0), 2);
            } else {
                continue;
            }

          double actual_diameter=0.13; 
          double focal_length=710.1; 
          int detected_diameter=total_x_pixel; 
          double distance=0;

          if(detected_diameter==0){
              distance=0 ;
          } else {
              distance=(actual_diameter*focal_length)/detected_diameter ;
          }

          distanceHistory.push_back(distance);

          // Apply moving average if we have enough data points
          if(distanceHistory.size() >= movingAverageWindow) { 
              std::vector<double> smoothedDistances=movingAverage(distanceHistory,movingAverageWindow); 

              // The latest smoothed distance is at the end of the vector
              double smoothedDistance=smoothedDistances.back(); 

              // Trend Analysis (Linear Extrapolation)
              double predictedDistance=0; 
              if(smoothedDistances.size() >= 2){ 
                  double currentSmoothed=smoothedDistances.back(); 
                  double previousSmoothed=smoothedDistances[smoothedDistances.size()-2]; 
                  double trend=currentSmoothed-previousSmoothed; 
                  predictedDistance = currentSmoothed+trend; 
              }

              char text[200]; 
              sprintf(text,"Distance: %.2f m (Smoothed: %.2f m Predicted: %.2f m)",distance ,smoothedDistance,predictedDistance); 
              cv::putText(frame,text ,cv::Point(0,40),cv::FONT_HERSHEY_SIMPLEX ,0.6 ,cv::Scalar(255 ,255 ,255),2 ); 

              printf( "Distance: %.2f m (Smoothed: %.2f m Predicted: %.2f m)\n",distance ,smoothedDistance,predictedDistance);
          } else { 
              char text[100]; 
              sprintf(text,"Distance: %.2f m",distance ); 
              cv::putText(frame,text ,cv::Point(x_new-r_new,y_new+r_new+40),cv::FONT_HERSHEY_SIMPLEX ,0.6 ,cv::Scalar(255 ,255 ,255),2 ); 
          }

         cv::line(frame ,cv::Point(center_x,center_y),cv::Point(x_new ,y_new),cv::Scalar(255 ,255 ,255),2 );

         break ; 
      } 
   } 

   return frame ; 
}
int main() {
    // Open video file or camera stream
    cv::VideoCapture cap("/home/abyan/Documents/testing_CVc++/ALTAIR_vision/samplesIMG/sample3.mp4");
    // VideoCapture cap(0, CAP_V4L2);

    if (!cap.isOpened()) {
        std::cout << "Error: No Video Opened" << std::endl;
        return -1;
    }

    // Uncommented VideoWriter code as in the original Python code
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter out("test2.mp4", fourcc, fps, cv::Size(width, height));
    
    // variables for measuring fps
    double currentFPS;
    double frameCount = 0;
    double startTime = static_cast<double>(cv::getTickCount());

    std::vector<double> distanceHistory; // Store the calculated distances
    int movingAverageWindow = 10; // Adjust the window size as needed

    while (true) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        if (!ret) {
            break;
        }

        frameCount++;

        // calculated elapsed time and fps
        double elapsedTime = (cv::getTickCount() - startTime) / cv::getTickFrequency();
        if (elapsedTime > 1.0) {
            currentFPS  = frameCount / elapsedTime;
            std::cout << "Current FPS: " << currentFPS << std::endl;

            // reset for next second
            frameCount = 0;
            startTime = static_cast<double>(cv::getTickCount());
        }

        int height = frame.rows;
        int width = frame.cols;

        center_x = width / 2;
        center_y = height / 2;

        int y_start = 0;
        int y_end = height;
        int x_start = 0;
        int x_end = width;

        // Draw vertical and horizontal lines on the frame
        cv::line(frame, cv::Point(center_x, y_start), cv::Point(center_x, y_end), cv::Scalar(255, 255, 255), 1);
        cv::line(frame, cv::Point(x_start, center_y), cv::Point(x_end, center_y), cv::Scalar(255, 255, 255), 1);

        // Process the field to get segmented field and its mask
        std::pair<cv::Mat, cv::Mat> fieldResult = field(frame);
        cv::Mat seg_field = fieldResult.first;
        cv::Mat mask_field = fieldResult.second;
        cv::Mat mask_ball = ball(seg_field);
        cv::Mat final_frame = detect(mask_ball, mask_field, frame, distanceHistory, movingAverageWindow);
        auto message = std_msgs::msg::String();
        message.data = "Hello, world! " + std::to_string(count_++);
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
        if (!final_frame.empty()) {
            
            // Overlay FPS on the frame
            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(currentFPS));
            cv::putText(final_frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);

            // Write the frame with overlay to output video
            out.write(final_frame);

            cv::imshow("ball detect", final_frame);
        }

        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    out.release();
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
