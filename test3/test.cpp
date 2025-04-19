#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Constants for merging and clustering
const double MAX_DISTANCE = 20.0;   // Maximum gap between lines to merge
const double MAX_ANGLE_DIFF = 15.0; // Maximum allowed angle difference

// --- DBSCAN Functions ---
double lineDistance(const Vec4i &l1, const Vec4i &l2) {
    double dist1 = hypot(l1[0] - l2[0], l1[1] - l2[1]);  // Start points distance
    double dist2 = hypot(l1[2] - l2[2], l1[3] - l2[3]);  // End points distance
    double angle1 = atan2(l1[3] - l1[1], l1[2] - l1[0]);
    double angle2 = atan2(l2[3] - l2[1], l2[2] - l2[0]);
    double angleDiff = fabs(angle1 - angle2);  // Angle difference
    return dist1 + dist2 + angleDiff * 10;  // Weighted distance
}

vector<vector<Vec4i>> clusterLines(vector<Vec4i> lines, double epsilon, int minPts) {
    vector<bool> visited(lines.size(), false);
    vector<vector<Vec4i>> clusters;
    
    for (size_t i = 0; i < lines.size(); ++i) {
        if (visited[i]) continue;
        visited[i] = true;

        vector<int> neighbors;
        for (size_t j = 0; j < lines.size(); ++j) {
            if (i != j && lineDistance(lines[i], lines[j]) < epsilon) {
                neighbors.push_back(j);
            }
        }

        if (neighbors.size() >= minPts) {
            vector<Vec4i> cluster = {lines[i]};
            for (int neighbor : neighbors) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    cluster.push_back(lines[neighbor]);
                }
            }
            clusters.push_back(cluster);
        }
    }
    return clusters;
}

// --- RANSAC Functions ---
double pointToLineDistance(Point2f pt, Vec4f line) {
    return abs((line[1] - line[3]) * pt.x - (line[0] - line[2]) * pt.y + line[0] * line[3] - line[1] * line[2]) /
           sqrt(pow(line[1] - line[3], 2) + pow(line[0] - line[2], 2));
}

Vec4f ransacLineFit(vector<Vec4i> lines, int iterations, double threshold) {
    Vec4f bestLine;
    int maxInliers = 0;

    for (int i = 0; i < iterations; ++i) {
        int idx1 = rand() % lines.size();
        int idx2 = rand() % lines.size();
        if (idx1 == idx2) continue;

        Point2f p1(lines[idx1][0], lines[idx1][1]);
        Point2f p2(lines[idx2][2], lines[idx2][3]);

        Vec4f model;
        fitLine(vector<Point2f>{p1, p2}, model, DIST_L2, 0, 0.01, 0.01);

        int inliers = 0;
        for (const auto &l : lines) {
            Point2f mid((l[0] + l[2]) / 2, (l[1] + l[3]) / 2);
            if (pointToLineDistance(mid, model) < threshold) {
                inliers++;
            }
        }

        if (inliers > maxInliers) {
            maxInliers = inliers;
            bestLine = model;
        }
    }
    return bestLine;
}

// --- Original Merging Functions ---
Point2f midpoint(const Vec4i &line) {
    return Point2f((line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0);
}

double length(const Vec4i &line) {
    return hypot(line[2] - line[0], line[3] - line[1]);
}

double slope(const Vec4i &line) {
    if (line[2] == line[0]) return DBL_MAX;
    return (double)(line[3] - line[1]) / (line[2] - line[0]);
}

double angleDifference(const Vec4i &X, const Vec4i &Y) {
    double thetaX = atan2(X[3] - X[1], X[2] - X[0]) * 180 / CV_PI;
    double thetaY = atan2(Y[3] - Y[1], Y[2] - Y[0]) * 180 / CV_PI;
    return fabs(thetaX - thetaY);
}

Point2f projectPointOntoLine(const Point2f &P, const Vec4i &line) {
    Point2f A(line[0], line[1]), B(line[2], line[3]);
    Point2f AP = P - A, AB = B - A;
    double t = (AP.x * AB.x + AP.y * AB.y) / (AB.x * AB.x + AB.y * AB.y);
    return A + t * AB;
}

Vec4i mergeSegments(const Vec4i &X, const Vec4i &Y) {
    Point2f Xm = midpoint(X);
    Point2f Ym = midpoint(Y);
    
    double lenX = length(X);
    double lenY = length(Y);
    double r = lenX / (lenX + lenY);
    
    Point2f P = r * Xm + (1 - r) * Ym;
    double theta = (lenX >= lenY) ? slope(X) : slope(Y);

    vector<Point2f> endpoints = { Point2f(X[0], X[1]), Point2f(X[2], X[3]),
                                  Point2f(Y[0], Y[1]), Point2f(Y[2], Y[3]) };
    vector<Point2f> projectedPoints;
    for (const auto &pt : endpoints) {
        projectedPoints.push_back(projectPointOntoLine(pt, X));
    }

    double min_proj = DBL_MAX, max_proj = -DBL_MAX;
    Point2f min_point, max_point;
    for (const auto &pt : projectedPoints) {
        double proj = (pt.x - P.x) * cos(theta) + (pt.y - P.y) * sin(theta);
        if (proj < min_proj) { min_proj = proj; min_point = pt; }
        if (proj > max_proj) { max_proj = proj; max_point = pt; }
    }
    return Vec4i((int)min_point.x, (int)min_point.y, (int)max_point.x, (int)max_point.y);
}

vector<Vec4i> mergeSimilarLines(vector<Vec4i> segments) {
    vector<Vec4i> mergedLines = segments;
    bool merged = true;
    
    while (merged) {
        vector<Vec4i> newLines;
        merged = false;
        
        for (size_t i = 0; i < mergedLines.size(); ++i) {
            bool mergedThis = false;
            for (size_t j = i + 1; j < mergedLines.size(); ++j) {
                if (length(mergedLines[i]) < 20 || length(mergedLines[j]) < 20) continue;

                if (angleDifference(mergedLines[i], mergedLines[j]) < MAX_ANGLE_DIFF &&
                    norm(Point(mergedLines[i][0], mergedLines[i][1]) - 
                         Point(mergedLines[j][0], mergedLines[j][1])) < MAX_DISTANCE) {
                    
                    newLines.push_back(mergeSegments(mergedLines[i], mergedLines[j]));
                    mergedLines.erase(mergedLines.begin() + j);
                    mergedThis = true;
                    merged = true;
                    break;
                }
            }
            if (!mergedThis) {
                newLines.push_back(mergedLines[i]);
            }
        }
        mergedLines = newLines;
    }
    return mergedLines;
}

int main() {
    Mat image = imread("soccer_field.jpg");
    Mat gray, edges;

    // Preprocessing
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Canny(gray, edges, 50, 150, 3);

    // Detect lines using Hough Transform
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);

    // Step 1: Cluster lines using DBSCAN
    double epsilon = 50.0;  // Adjust based on your image scale
    int minPts = 2;         // Minimum points for a cluster
    vector<vector<Vec4i>> clusteredLines = clusterLines(lines, epsilon, minPts);

    // Step 2: Process each cluster with RANSAC and merge
    vector<Vec4i> finalLines;
    for (const auto &cluster : clusteredLines) {
        // Apply RANSAC to fit a dominant line
        Vec4f bestLine = ransacLineFit(cluster, 100, 10.0);

        // Convert RANSAC line to Vec4i format (approximate endpoints)
        Point2f p1(bestLine[2], bestLine[3]);  // Point on the line
        Point2f dir(bestLine[0], bestLine[1]); // Direction vector
        double tMin = -1000, tMax = 1000;      // Arbitrary range for visualization
        Point2f start = p1 + tMin * dir;
        Point2f end = p1 + tMax * dir;
        Vec4i ransacLine((int)start.x, (int)start.y, (int)end.x, (int)end.y);

        // Merge lines within the cluster
        vector<Vec4i> clusterLines = cluster;
        clusterLines.push_back(ransacLine); // Include RANSAC line in merging
        vector<Vec4i> mergedCluster = mergeSimilarLines(clusterLines);

        finalLines.insert(finalLines.end(), mergedCluster.begin(), mergedCluster.end());
    }

    // Draw results
    Mat result = image.clone();
    for (const auto &l : lines) {
        line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 0, 0), 1);
    }
    for (const auto &l : finalLines) {
        line(result, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
    }

    imshow("Original Lines", image);
    imshow("Final Lines (DBSCAN + RANSAC + Merging)", result);
    waitKey(0);
    return 0;
}