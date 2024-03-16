//
// Created by yao on 29/05/18.
//

#pragma once
#include <memory>
#include <opencv2/core.hpp>
class gms_matcher;

class robust_gms_matcher_t {
public:
    robust_gms_matcher_t(const std::vector<cv::KeyPoint> &vkp1, const cv::Size size1, const std::vector<cv::KeyPoint> &vkp2, const cv::Size size2, const std::vector<cv::DMatch> &vDMatches,bool pre_rotation = false);
    ~robust_gms_matcher_t();
    int GetInlierMask(std::vector<bool> &vbInliers, bool WithScale = false, bool WithRotation = false);
private:
    std::unique_ptr<gms_matcher> _matcher;
};