#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "circledec.h"
#include <time.h>
#include <codecvt>

extern int rows;
extern int cols;

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

void test_batch(string dir_path) {
    namespace fs = std::filesystem;
    for (const auto &entry: fs::directory_iterator(dir_path)) {
        u8string u8path_string{entry.path().u8string()};
        u8string u8filename{entry.path().filename().u8string()};
        string path_string(u8path_string.cbegin(), u8path_string.cend());
        string filename(u8filename.cbegin(), u8filename.cend());
        cv::Mat src = cv::imread(path_string, cv::IMREAD_GRAYSCALE);
        assert(!src.empty());
        rows = src.rows;
        cols = src.cols;
        vector<CircleType> circles;
        vector<vector<cv::Point2f>> contours;
        int ret = detect(src, contours, circles);
        if (ret == -1) {
            cout << "Cann't find contours in file " << filename << endl;
            continue;
        } else {
            cout << "Found contours in file " << filename << endl;
        }

#ifdef DEBUG
//        cout << "Time of total: " << double(end - start) / CLOCKS_PER_SEC << endl;
        for (int i = 0; i < circles.size(); i++) {
            cv::Vec3f c = circles[i];
            cv::Point center(c[0], c[1]);
            cv::circle(src, center, c[2], cv::Scalar(255, 255, 255), cv::FILLED);
        }
        cv::imwrite(filename, src);
#endif
    }
}

int main(int argc, char **argv) {
    test_batch("C:\\Users\\syemc\\source\\repos\\pose-emti-v2\\c4x2000");

    return 0;
}
