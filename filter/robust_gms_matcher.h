/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

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