#pragma once
#ifndef __CIRCLEDEC_H_
#define __CIRCLEDEC_H_

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

typedef cv::Vec3f CircleType;	// float x,y,r

inline double getRoundness(double area,double arcLen) {
	return std::abs(1 - ((4 * CV_PI * area) / (arcLen * arcLen)));
}

/*
* 画直方图，统计用（调试）
*/
void drawHist(cv::Mat& src,cv::Mat &dst) {

}

/*
* 自动gamma矫正
*/
void autoGamma(cv::Mat& src, cv::Mat& dst) {
	const int channels = src.channels();
	const int type = src.type();
	assert(type == CV_8U);	// 只限8bit

	// 计算中位数
	cv::Scalar mean_scalar = cv::mean(src);
	double mean = mean_scalar[0];

	// 某篇论文的gamma value设置值
	double gamma_value = std::log10(0.5) / std::log10(mean / 255);
	
	// 归一化，然后gamma变换
	cv::Mat norm,gamma;
	cv::normalize(src, norm, 1.0, 0.0, cv::NORM_MINMAX,CV_64F);
	//src.convertTo(norm,CV_64F,1/255.0);
	cv::pow(norm, gamma_value, gamma);

	cv::convertScaleAbs(gamma, dst, 255.0);
}

/*
* 预处理，包含一些图像增强方法
*/
void pretreatment(cv::Mat &src, cv::Mat &dst) {
	cv::Mat gray, bin, equalize;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::imwrite("gray.jpg", gray);
	
	// 直方图均衡算法
	cv::equalizeHist(gray, equalize);
	cv::imwrite("equalize.jpg", equalize);

	// 二值化: OTSU
	//cv::threshold(equalize, bin, 24, 255, cv::THRESH_OTSU);
	//cv::imwrite("otsu.jpg", bin);

	// 二值化，自适应，先用该算法试试
	cv::Mat adaptBin;
	cv::adaptiveThreshold(equalize, adaptBin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 131,21);
	cv::bitwise_not(adaptBin, adaptBin);
	cv::imwrite("adaptiveBinary.jpg", adaptBin);

	// 椒盐去噪(中值滤波)
	cv::Mat blur;
	cv::medianBlur(adaptBin, dst, 5);
	cv::imwrite("blur.jpg", dst);

	// 闭运算
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//cv::morphologyEx(adaptBin, dst, cv::MORPH_CLOSE, kernel);
	//cv::imwrite("close.jpg", dst);
}

void fitCircle(std::vector<cv::Point> &contour,CircleType &c) {
	cv::RotatedRect rect = cv::fitEllipse(contour);
	cv::Point2f center = rect.center;
	float r = (rect.size.width + rect.size.height) / 4;
	c[0] = center.x;
	c[1] = center.y;
	c[2] = r;
}

/*
* 通过轮廓找到圆
* src:		通过图像处理后的图片,二值化图片
* circles:	找到的疑似圆形	
*/
void findCircleByContours(cv::Mat &src,std::vector<CircleType> &circles) {	// 通过轮廓找到圆
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy; // 树形结构层次关系
	cv::findContours(src, contours,hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
	//cv::drawContours()
	cv::Mat pix(src.rows, src.cols, CV_8UC3,cv::Scalar(0,0,0));
	double area, arcLen;
	for (int i = 0; i < contours.size(); i++) {
		area = cv::contourArea(contours[i]);
		arcLen = cv::arcLength(contours[i], true);

		if (area < 50) {
			continue;
		}
		double roundness = getRoundness(area, arcLen);
		if (roundness > 0.175) {
			continue;
		}
		CircleType c;
		fitCircle(contours[i], c);
		circles.push_back(c);
		cv::drawContours(pix, contours, i, cv::Scalar(255, 255, 255));
		std::cout << i << ": " << area << ", " << roundness << std::endl;
	}
	cv::imwrite("contours.jpg", pix);
}


void findCircleByConnectedComponents(cv::Mat src,std::vector<CircleType> &circles) {
	//pretreatment(src, );
	cv::Mat labels, stats, centroids;
	int n_components = cv::connectedComponentsWithStats(src, labels, stats, centroids);	// 连通域
	for (int i = 0; i < n_components; i++) {
		int x = stats.at<int>(i, cv::CC_STAT_LEFT);
		int y = stats.at<int>(i, cv::CC_STAT_TOP);
		int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);
		int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
		int area = stats.at<int>(i, cv::CC_STAT_AREA);
		//std::cout << i << ":" << area << std::endl;
		//cv::arcLength()
	}
	
}

#endif // __CIRCLEDEC_H_
