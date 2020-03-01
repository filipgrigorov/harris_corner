#pragma once
#include <vector>

#include <opencv2/opencv.hpp>

std::vector<cv::Point> FindHarris(const cv::Mat& image, int ksize, float k);
