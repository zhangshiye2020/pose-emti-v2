#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "circledec.h"

using namespace std;
int main(int argc, char** argv) {
	cv::Mat mat = cv::imread("C:\\Users\\zhang\\Pictures\\Image_20210929204316983.bmp");
	//cout << mat.channels() << endl;
	cv::Mat dst,pre_src;
	pretreatment(mat, pre_src);
	vector<CircleType> circles;
	findCircleByContours(pre_src, circles);
	for (int i = 0; i < circles.size(); i++) {
		cv::Vec3f c = circles[i];
		cv::Point center(c[0],c[1]);
		cv::circle(mat, center, c[2], cv::Scalar(255, 255, 255), cv::FILLED);
	}
	cv::imwrite("circle.jpg", mat);
	return 0;
}