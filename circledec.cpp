#include "circledec.h"

/*
 * gamma矫正
 */
void gammaCorrection(cv::Mat &src, cv::Mat &dst, double gamma) {
    int channels = src.channels();
    assert(channels == 1);
    unsigned char lut_data[256];
    for (int i = 0; i < 256; i++) {
        lut_data[i] = cv::saturate_cast<uchar>(pow((float) i / 255, gamma) * 255.0f);
    }
    cv::Mat lut(1, 256, CV_8U, lut_data);
    cv::LUT(src, lut, dst);
}

/*
* 自动gamma矫正，不好用，损失太多细节
*/
void autoGamma(cv::Mat &src, cv::Mat &dst) {
    const int channels = src.channels();
    const int type = src.type();
    assert(type == CV_8U || type == CV_8UC3);    // 只限8bit

    // 计算中位数
    cv::Scalar mean_scalar = cv::mean(src);
    double mean = mean_scalar[0];

    // 某篇论文的gamma value设置值
    double gamma_value = std::log10(0.5) / std::log10(mean / 255);
//    GammaCorrection(src, dst, gamma_value);

//    double gamma_value = 0.3;
    // 归一化，然后gamma变换
    cv::Mat norm, gamma;
    cv::normalize(src, norm, 1.0, 0.0, cv::NORM_MINMAX, CV_64F);
    //src.convertTo(norm,CV_64F,1/255.0);
    cv::pow(norm, gamma_value, gamma);

    cv::convertScaleAbs(gamma, dst, 255.0);


#ifdef DEBUG
    cv::imwrite("gamma.jpg", dst);
#endif

}

/*
* 预处理，包含一些图像增强方法，主要包括直方图均衡，二值化自适应，中值滤波
*/
void pretreatment(cv::Mat &src, cv::Mat &dst) {
    clock_t start = clock();
    cv::Mat gray, bin, equalize;
//    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 直方图均衡算法
//    clock_t start_hist = clock();
//    cv::equalizeHist(src, equalize);
//    clock_t end_hist = clock();
//    std::cout << "Time of hist: " << double(end_hist - start_hist) / CLOCKS_PER_SEC << std::endl;
    // gamma矫正
    cv::Mat gamma;
    gammaCorrection(src, gamma, 0.5);
    // 
//    autoGamma(gray, gamma);

    // 二值化，自适应，先用该算法试试
    clock_t start_adapt = clock();
    cv::Mat adaptBin;
    //cv::threshold(gamma, adaptBin, 128, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
    cv::adaptiveThreshold(gamma, adaptBin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 191, 21);

    clock_t end_adapt = clock();
    std::cout << "Time of adaptive: " << double(end_adapt - start_adapt) / CLOCKS_PER_SEC << std::endl;

    // 去除黑点，闭运算，去除毛点，白点
    clock_t start_co = clock();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat close, open;
    cv::morphologyEx(adaptBin, close, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(close, dst, cv::MORPH_OPEN, kernel);
    clock_t end_co = clock();
    std::cout << "Time of Close/Open: " << double(end_co - start_co) / CLOCKS_PER_SEC << "s" << std::endl;
//
    clock_t end = clock();

#ifdef DEBUG
    std::cout << "Time of pretreatment: " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
#endif
#ifdef DEBUG
    //    cv::imwrite("gray.jpg", gray);
    cv::imwrite("gamma.jpg", gamma);
    cv::imwrite("adaptiveBinary.jpg", adaptBin);
    cv::imwrite("blur.jpg", dst);
#endif
}

/*
 * 从轮廓中拟合圆，算法粗糙
 * contour:vector<Point> 圆的轮廓
 * c:CircleType-Vec3f    圆
 */
void fitCircle(std::vector<cv::Point> &contour, CircleType &c) {
    cv::RotatedRect rect = cv::fitEllipse(contour);
    cv::Point2f center = rect.center;
    float r = (rect.size.width + rect.size.height) / 4;
    c[0] = center.x;
    c[1] = center.y;
    c[2] = r;
}

/*
 * 检测，总程序封装
 */
void detect(cv::Mat &im, std::vector<std::vector<cv::Point2f>> &contours_f, std::vector<CircleType> &circles) {
    cv::Mat pre_src;
    pretreatment(im, pre_src);
    std::vector<std::vector<cv::Point>> contours;
    findCircleByContours(pre_src, contours, circles);
    std::sort(circles.begin(), circles.end(), centerCmp);

    for (int i = 0; i < contours.size(); i++) {
        std::vector<cv::Point2f> ctr_f;
        cv::Mat(contours[i]).convertTo(ctr_f, cv::Mat(ctr_f).type());
        contours_f.push_back(ctr_f);
    }
}

/*
 * 计算轮廓中心
 */
void getCenterFromContours(std::vector<cv::Point> &contour, cv::Point2f &center) {
    cv::RotatedRect rect = cv::fitEllipse(contour);
    center = rect.center;
}

void getCenterFromContours(std::vector<std::vector<cv::Point2f>> &contour, std::vector<cv::Point2f> &center) {
    for (int i = 0; i < contour.size(); i++) {
        cv::RotatedRect rect = cv::fitEllipse(contour[i]);
        center.push_back(rect.center);
    }
}

/*
* 圆心满足3r-5r关系
*/
bool near_point(cv::Point2f p1, cv::Point2f p2, float r) {
    float dis = std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    float min_dis_err = (4 - THRESH_DIS) * r;
    float max_dis_err = (4 + THRESH_DIS) * r;
    if (dis < min_dis_err || dis > max_dis_err) {   // 满足
        return false;
    }
//    float dis_x = std::abs(p1.x - p2.x);  //假设同y
//    float dis_y = std::abs(p1.y - p2.y);    //假设同x
//    float min_line_err = dis - THRESH_LINE_R * r;   // 最小距离 dis - 0.5r
//    float max_line_err = dis + THRESH_LINE_R * r;   // 最大距离 dis + 0.5r
//    if ((dis_x < min_line_err || dis_x > max_line_err) && (dis_y < min_line_err || dis_y > max_line_err)) {
//        std::cout << "exe times: " << std::endl;
//        return false;
//    }
    return true;
}
/*
 * 点是否可以与其他点组成直线???
 */
void in_line(std::vector<cv::Point2f> points) {

}

void filterContoursCore(std::vector<cv::Point2f> &mc, std::vector<float> &radio_v, std::vector<int> &contours_index,
                        std::vector<int> &circles_index) {
    // 允许误差 3.5r-4.5r之间
    bool related; // 有关系，表示满足near_and_in_line关系
    int length = mc.size();
    std::vector<std::vector<bool>> related_map(length, std::vector<bool>(length));
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < length; j++) {
            if (i == j) {
                related_map[i][j] = false;
                continue;
            }
            cv::Point2f p1 = mc[i];
            cv::Point2f p2 = mc[j];
            float r1 = radio_v[i];
            float r2 = radio_v[j];
            if (r1 / r2 < 0.7 / 1 || r1 / r2 > 1 / 0.7) {  // 两个 r 相差太大
                related_map[i][j] = false;
                continue;
            }
            related = near_point(p1, p2, (r1 + r2) / 2);
            related_map[i][j] = related;
        }
    }

    std::vector<bool> vitis(length, false);

    for (int i = 0; i < length; i++) {
        circles_index.clear();
        vitis[i] = true;
        int sum = std::accumulate(related_map[i].begin(), related_map[i].end(), 0); // 连接数量
        if (sum == 0) {
            continue;
        }
        std::queue<int> related_q;
        related_q.push(i);
        int cnt = 0;
        circles_index.push_back(contours_index[i]);
        while (!related_q.empty()) {
            int root = related_q.front();
            related_q.pop();
            for (int j = 0; j < length; j++) {
                if (!vitis[j] && related_map[root][j]) {
                    cnt++;
                    related_q.push(j);
                    vitis[j] = true;
                    circles_index.push_back(contours_index[j]);
                }
            }
        }
        if (circles_index.size() == CIRCLE_NUM) {
            return;
        }
    }
}

/*
 * 过滤无效的边缘，返回轮廓下标，如果不存在就返回 -1，存在返回个数
 * 主要通过轮廓的面积(> 50)、圆度、以及结构关系判断
 * contours:   边缘
 * hierarchy:  轮廓之间的树形关系
 * contoursIndex: 圆形轮廓下标
 */
int filterContours(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Vec4i> &hierarchy,
                   std::vector<int> &circles_index) {
    // todo: 算法存在缺陷，需要添加相对位置限制，相对面积限制
    // todo: 在父节点下这一招未必好使
    int ret = -1;
    clock_t start = clock();
    std::unordered_map<int, std::vector<int>> map_index;   // 父子关系图
    std::unordered_map<int, std::vector<double>> map_areas;  // 同一个父节点下所有轮廓的面积
    std::unordered_map<int, std::vector<float>> map_arcLen;
    std::vector<int> keys;
    double area, arcLen;
    for (int i = 0; i < contours.size(); i++) { // 粗过滤
        area = cv::contourArea(contours[i]);
        arcLen = cv::arcLength(contours[i], true);

        if (area < THRESH_MIN_AREA) {
            continue;
        }
        double roundness = GET_ROUNDNESS(area, arcLen);
        if (roundness > THRESH_ROUNDNESS) {
            continue;
        }
        int p = hierarchy[i][3];
        map_index[p].push_back(i);
        map_areas[p].push_back(area);
        map_arcLen[p].push_back(arcLen);
        keys.push_back(p);
    }

    // 一群轮廓里面找圆
#ifdef DEBUG
    cv::Mat first_filter_contours(819, 901, CV_8UC1, cv::Scalar(0, 0, 0));
    for (auto iter = map_index.begin(); iter != map_index.end(); iter++) {
        for (int i = 0; i < iter->second.size(); i++) {
            cv::drawContours(first_filter_contours, contours, (iter->second)[i], cv::Scalar(255, 255, 255));
            //cv::putText(first_filter_contours, std::to_string(iter->first), cv::Point(mc[i].x, mc[i].y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
        }
        cv::imwrite("first_filter_contours.jpg", first_filter_contours);
    }
#endif // DEBUG

    for (auto iter = map_index.begin(); iter != map_index.end(); iter++) {  // 将所有轮廓按照父轮廓分割成x组
        int p = iter->first;
        if (iter->second.size() < 9) {  // 至少9个轮廓
            continue;
        }
        std::vector<std::vector<cv::Point>> contours_it;
        for (int i = 0; i < iter->second.size(); i++) {
            contours_it.push_back(contours[iter->second[i]]);
        }
        std::vector<float> radio_v(map_arcLen.at(p).size());
        for (int i = 0; i < iter->second.size(); i++) {
            radio_v[i] = map_arcLen.at(p)[i] / CV_2PI;
        }
        std::vector<cv::Point2f> mc;
        getMcOfContours(contours_it, mc);

        filterContoursCore(mc, radio_v, map_index.at(p), circles_index); // 一组轮廓中尝试提取出CIRCLE_NUM个圆
        if (circles_index.size() == CIRCLE_NUM) {   // 说明找到了
#ifdef DEBUG
            cv::Mat second_filter_contours(819, 901, CV_8UC1, cv::Scalar(0, 0, 0));
            for (auto iter = circles_index.begin(); iter != circles_index.end(); iter++) {
                cv::drawContours(second_filter_contours, contours, *iter, cv::Scalar(255, 255, 255));
                cv::imwrite("second_filter_contours.jpg", second_filter_contours);
            }
            std::cout << "Find Target, Parent = " << p << std::endl;
#endif // DEBUG
            return CIRCLE_NUM;
        }
    }

    return -1;
}

/**
* 计算轮廓中心
*/
void getMcOfContours(std::vector<std::vector<cv::Point>> &contours, std::vector<cv::Point2f> &mc) {
    std::vector<cv::Moments> mu(contours.size());
    mc.assign(contours.size(), cv::Point2f(0, 0));
    for (int i = 0; i < contours.size(); i++) {
        mu[i] = cv::moments(contours[i], false);
        mc[i] = cv::Point2d(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }
}

/*
* 通过轮廓找到圆
* src:		通过图像处理后的图片,二值化图片
* circles:	圆形
*/
void findCircleByContours(cv::Mat &src, std::vector<std::vector<cv::Point>> &contours_circle,
                          std::vector<CircleType> &circles) {    // 通过轮廓找到圆
    std::vector<std::vector<cv::Point>> contours;
    clock_t start = clock();
//    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy; // 树形结构层次关系
    cv::findContours(src, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);
    clock_t end = clock();
#ifdef DEBUG
    std::cout << "Time of find contours: " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
    cv::Mat picFul(src.rows, src.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::drawContours(picFul, contours, -1, cv::Scalar(255, 255, 255));
    cv::imwrite("contoursFull.jpg", picFul);
#endif

    std::vector<int> contoursIndex;
    int ret = filterContours(contours, hierarchy, contoursIndex); // 圆形轮廓下标

    if (ret == -1) {    // 没找到匹配的目标
        std::cout << "Can't find Contours" << std::endl;
        return;
    }
    // 轮廓
//    std::vector<std::vector<cv::Point>> circleContours;
    clock_t startOfFitCircle = clock();
    CircleType c;
    std::vector<cv::Point> circleContour;
    for (int i = 0; i < contoursIndex.size(); i++) {
        int cIndex = contoursIndex[i];
        circleContour = contours[cIndex];
        fitCircle(circleContour, c);
        circles.push_back(c);
        contours_circle.push_back(circleContour);
    }
    clock_t endOfFitCircle = clock();
#ifdef DEBUG
    std::cout << "Time of fit circle: " << double(endOfFitCircle - startOfFitCircle) / CLOCKS_PER_SEC << "s"
              << std::endl;
#endif

#ifdef DEBUG
    cv::Mat pic(src.rows, src.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int i = 0; i < contoursIndex.size(); i++) {
        int cIndex = contoursIndex[i];
        cv::drawContours(pic, contours, cIndex, cv::Scalar(255, 255, 255));
    }
    cv::imwrite("contours.jpg", pic);
#endif
}

/*
 * 已废弃，该连通域函数结果无法计算出其周长
 */
void findCircleByConnectedComponents(cv::Mat src, std::vector<CircleType> &circles) {
    //pretreatment(src, );
    cv::Mat labels, stats, centroids;
    int n_components = cv::connectedComponentsWithStats(src, labels, stats, centroids);    // 连通域
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

/*
 * 排序逻辑
 */
bool centerCmp(CircleType &c1, CircleType &c2) {
    // 同 y
    float dy = c1[1] - c2[1];
    float dx = c1[0] - c2[0];
    float r = (c1[2] + c2[2]) / 4;  // 1/2半径
    float bias = THRESH_LINE_R * r;
//    return dy<-5?true:dy>5?dx>5:false:false
    if (-bias > dy) {
        return true;
    } else if (dy > bias) {
        return false;
    }
    if (dx > bias) {
        return false;
    } else if (dx < -bias) {
        return true;
    }
    return false;
}

void sortCircles(std::vector<CircleType> &circles) {
    std::sort(circles.begin(), circles.end(), centerCmp);
}
