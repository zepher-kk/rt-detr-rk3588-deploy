#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

struct DetectResult {
    int class_id;
    float score;
    cv::Rect box;
};

constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr int NUM_CLASSES = 10;

const std::vector<std::string> CLASSES = {
    "Pedestrian", "People", "Bicycle", "Car", "Van", 
    "Truck", "Tricycle", "Awning-tricycle", "Bus", "Motor"
};