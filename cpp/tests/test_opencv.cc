#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 打印完整的构建信息
    std::cout << cv::getBuildInformation() << std::endl;
    return 0;
}

