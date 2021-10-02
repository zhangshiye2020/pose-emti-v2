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
* ��ֱ��ͼ��ͳ���ã����ԣ�
*/
void drawHist(cv::Mat& src,cv::Mat &dst) {

}

/*
* �Զ�gamma����
*/
void autoGamma(cv::Mat& src, cv::Mat& dst) {
	const int channels = src.channels();
	const int type = src.type();
	assert(type == CV_8U);	// ֻ��8bit

	// ������λ��
	cv::Scalar mean_scalar = cv::mean(src);
	double mean = mean_scalar[0];

	// ĳƪ���ĵ�gamma value����ֵ
	double gamma_value = std::log10(0.5) / std::log10(mean / 255);
	
	// ��һ����Ȼ��gamma�任
	cv::Mat norm,gamma;
	cv::normalize(src, norm, 1.0, 0.0, cv::NORM_MINMAX,CV_64F);
	//src.convertTo(norm,CV_64F,1/255.0);
	cv::pow(norm, gamma_value, gamma);

	cv::convertScaleAbs(gamma, dst, 255.0);
}

/*
* Ԥ��������һЩͼ����ǿ����
*/
void pretreatment(cv::Mat &src, cv::Mat &dst) {
	cv::Mat gray, bin, equalize;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	cv::imwrite("gray.jpg", gray);
	
	// ֱ��ͼ�����㷨
	cv::equalizeHist(gray, equalize);
	cv::imwrite("equalize.jpg", equalize);

	// ��ֵ��: OTSU
	//cv::threshold(equalize, bin, 24, 255, cv::THRESH_OTSU);
	//cv::imwrite("otsu.jpg", bin);

	// ��ֵ��������Ӧ�����ø��㷨����
	cv::Mat adaptBin;
	cv::adaptiveThreshold(equalize, adaptBin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 131,21);
	cv::bitwise_not(adaptBin, adaptBin);
	cv::imwrite("adaptiveBinary.jpg", adaptBin);

	// ����ȥ��(��ֵ�˲�)
	cv::Mat blur;
	cv::medianBlur(adaptBin, dst, 5);
	cv::imwrite("blur.jpg", dst);

	// ������
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//cv::morphologyEx(adaptBin, dst, cv::MORPH_CLOSE, kernel);
	//cv::imwrite("close.jpg", dst);
}

void fitCircle(std::vector<cv::Point> &contour,CircleType &c) {
	cv::RotatedRect rect = cv::fitEllipse(contour);
	cv::Point2f center = rect.center;
	float r = (rect.size.width + rect.size.height) / 2;
	c[0] = center.x;
	c[1] = center.y;
	c[2] = r;
}

/*
* ͨ�������ҵ�Բ
* src:		ͨ��ͼ������ͼƬ,��ֵ��ͼƬ
* circles:	�ҵ�������Բ��	
*/
void findCircleByContours(cv::Mat &src,std::vector<CircleType> &circles) {	// ͨ�������ҵ�Բ
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy; // ���νṹ��ι�ϵ
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
	int n_components = cv::connectedComponentsWithStats(src, labels, stats, centroids);	// ��ͨ��
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
