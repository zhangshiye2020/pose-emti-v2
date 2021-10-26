#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "circledec.h"
#include <time.h>

using namespace std;

/*
 * 聚类算法分类，但是效果不一定好
 */
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
    clock_t start = clock();
    string folder = "../c4/";
    string filename = "Image_20211025152141212";
    string fileExtension = ".bmp";

    cv::Mat gray;
    cv::Mat mat = cv::imread(folder + filename + fileExtension, cv::IMREAD_GRAYSCALE);
    assert(!mat.empty());
    rows = mat.rows;
    cols = mat.cols;
//    cv::imshow("gra",mat);
//    cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
    cv::Mat dst, pre_src;
//    pretreatment(mat, pre_src);
    vector<CircleType> circles;
    vector<vector<cv::Point2f>> contours;
//    findCircleByContours(pre_src, circles);

    detect(mat, contours, circles);
    clock_t end = clock();
    cout << "Time of total: " << double(end - start) / CLOCKS_PER_SEC << endl;

#ifdef DEBUG
    cout << "Time of total: " << double(end - start) / CLOCKS_PER_SEC << endl;
    for (int i = 0; i < circles.size(); i++) {
        cv::Vec3f c = circles[i];
        cv::Point center(c[0], c[1]);
        cv::circle(mat, center, c[2], cv::Scalar(255, 255, 255), cv::FILLED);
    }
    cv::imwrite(filename + ".bmp", mat);
#endif

    return 0;
}
