﻿#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "circledec.h"
#include <time.h>

using namespace std;

void filterContoursByArea(cv::Mat contourAreas, int classes) {
    // 聚类算法
    cv::Mat labels;
    cv::kmeans(contourAreas, classes, labels,
               cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 10, 0), 5, cv::KMEANS_PP_CENTERS);

    for (int i = 0; i < labels.rows; i++) {
        cout << " " << labels.at<int>(i, 0);
    }
    cout << endl;
    for (int i = 0; i < contourAreas.rows; i++) {
        cout << " " << contourAreas.at<float>(i, 0);
    }
}

int main(int argc, char **argv) {
//    cv::Mat mat = cv::imread("C:\\Users\\zsy\\Documents\\GitHub\\pose-emti\\Image_20210929204354303.bmp");
    cv::Mat mat;
    const string filedir = "C:\\Users\\zsy\\Pictures\\";
    const vector<string> filename_v = {"Image_20210929204244455", "Image_20210929204307167", "Image_20210929204316983",
                                       "Image_20210929204325775", "Image_20210929204344239", "Image_20210929204354303"};
//    for(int i=0;i<filename_v.size())
    for (auto filename: filename_v) {
        clock_t start = clock();    // 计时
        cout << "Detecting image :" << filename << endl;
        string filepath = filedir + filename;
        mat = cv::imread(filepath + ".bmp");
        cv::Mat dst, pre_src;
        pretreatment(mat, pre_src);
        vector<CircleType> circles;
        findCircleByContours(pre_src, circles);
        clock_t end = clock();
        cout << "Time of total: " << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
#ifdef _DEBUG
        for (int i = 0; i < circles.size(); i++) {
            cv::Vec3f c = circles[i];
            cv::Point center(c[0], c[1]);
            cv::circle(mat, center, c[2], cv::Scalar(255, 255, 255), cv::FILLED);
        }
        cv::imwrite(filename + ".bmp", mat);
#endif
    }



//    vector<float> v{50.5, 56.5, 1031.5, 1059, 8021.5, 84, 93.5, 93.5, 88, 93, 88, 84, 88.5, 86.5, 55.5};
//    cv::Mat mat(v);
//    filterContoursByArea(mat, 4);

    return 0;
}