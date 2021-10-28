#ifndef __CIRCLEDEC_H_
#define __CIRCLEDEC_H_

/*
 * zhangshiye, 2021-9
 * 圆度：程序中圆度定义与外界不一样，为 1 - roundness
 */

#define DEBUG
#define CIRCLE_NUM 4
#define THRESH_MIN_AREA 50     // 最小面积要求
#define THRESH_ROUNDNESS 0.3   // 圆度要求，大于该圆度的统统不算
#define THRESH_MAZ_ERR   1000  // 面积差异的距离，这个未必有用
#define GET_ROUNDNESS(area, arcLen) ( 1 - ((4 * CV_PI * (area))/((arcLen) * (arcLen))))
#define THRESH_LINE_R  0.5       //  圆心是否在同一行上所允许的误差，允许误差多少个r
//#define sqrt2 1.414213562373095
#define THRESH_DIS  1.3   // 圆心间隔的距离，误差最多 1r

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <unordered_map>
#include <time.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>

extern int rows;
extern int cols;

typedef cv::Vec3f CircleType;    // float x,y,r

void
findCircleByContours(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours, std::vector<CircleType> &circles);

void gammaCorrection(cv::Mat &src, cv::Mat &dst, double gamma);

void pretreatment(cv::Mat &src, cv::Mat &dst);

void fitCircle(std::vector<cv::Point> &contour, CircleType &c);

int detect(cv::Mat &im, std::vector<std::vector<cv::Point2f>> &contours_f, std::vector<CircleType> &circles);

void getCenterFromContours(std::vector<cv::Point> &contour, cv::Point2f &center);

void filterContoursCore(std::vector<cv::Point2f> &mc, std::vector<float> &radio_v, std::vector<int> &contours_index,
                        std::vector<int> &circles_index);

int filterContours(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i> &hierarchy,
                   std::vector<int> &circles_index);

bool centerCmp(CircleType &c1, CircleType &c2);

void sortCircles(std::vector<CircleType> &centers);

void getMcOfContours(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &mc);

bool near_point(cv::Point2f p1, cv::Point2f p2, float r);

void BFSTrace(std::vector<std::vector<bool>> &related_map, std::vector<int> &contours_index,
              std::vector<int> &circles_index, int length);

inline double getRoundness(double area, double arcLen) {    // 圆度似乎与正余弦有关，永远不可能大于1
    return 1 - ((4 * CV_PI * area) / (arcLen * arcLen));
}

void getCenterFromContours(std::vector<std::vector<cv::Point2f>> &contour, std::vector<cv::Point2f> &center);

#endif // __CIRCLEDEC_H_
