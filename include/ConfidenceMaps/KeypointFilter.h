#ifndef KEYPOINT_FILTER_H
#define KEYPOINT_FILTER_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>  // Include the PyTorch header

// Declare the filtering function
void FilterKeyPoints(std::vector<cv::KeyPoint>& _keypoints, cv::Mat& descriptors, cv::Mat image, const std::string& confidenceLevel, std::optional<float> threshold);

#endif