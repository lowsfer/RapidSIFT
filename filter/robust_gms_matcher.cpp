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

#include <random>
#include "robust_gms_matcher.h"
#include "gms_matcher.h"
#include <cmath>

//@fixme: add unit test
//@fixme: use ransac/procas affine2 or sim2 rotation instead of this hacky method
std::pair<float, float> find_rotation_scale(const std::vector<cv::KeyPoint> &vkp1, const std::vector<cv::KeyPoint> &vkp2, const std::vector<cv::DMatch> &vDMatches, int min_length = 50, size_t num_buckets = 64)
{
    std::vector<cv::DMatch> matches = vDMatches;
    std::partial_sort(matches.begin(), matches.begin() + vDMatches.size() / 4, matches.end(), [](const cv::DMatch& a, const cv::DMatch& b){return a.distance < b.distance;});
    matches.resize(vDMatches.size() / 4);
    if(matches.size() < 2)
        return std::make_pair(0.f, 1.f);

    std::vector<size_t> buckets(num_buckets, 0u);
    std::default_random_engine rng{std::random_device{}()};
    std::uniform_int_distribution<size_t> dist(0, matches.size() - 1);
//    const int sqr_min_length = min_length * min_length;
#ifndef NDEBUG
    cv::Point2f l0, l1, r0, r1;
#endif
    auto get_angle_scale = [&]() -> std::pair<float, float> {
        cv::Point2f arrow_left, arrow_right;
        for(int i = 0; i < 32; i++){
            const cv::DMatch &m0 = matches[dist(rng)];
            const cv::DMatch &m1 = matches[dist(rng)];
#ifndef NDEBUG
            l0 = vkp1[m0.queryIdx].pt; l1 = vkp1[m1.queryIdx].pt;
            r0 = vkp2[m0.trainIdx].pt; r1 = vkp2[m1.trainIdx].pt;
#endif
            arrow_left = vkp1[m0.queryIdx].pt - vkp1[m1.queryIdx].pt;
            arrow_right = vkp2[m0.trainIdx].pt - vkp2[m1.trainIdx].pt;
            const float length_left = std::sqrt(float(arrow_right.x * arrow_right.x + arrow_right.y * arrow_right.y));
            const float length_right = std::sqrt(float(arrow_left.x * arrow_left.x + arrow_left.y * arrow_left.y));
            if(std::abs(length_left - length_right) > 50 || length_left < min_length || length_right < min_length)
                continue;

            float angle = std::atan2(
                    arrow_left.x * arrow_right.y - arrow_left.y * arrow_right.x,
                    arrow_left.x * arrow_right.x - arrow_left.y * arrow_right.y);
            if(angle < 0)
                angle += float(M_PI * 2);
            const float scale = std::sqrt(float(arrow_right.x * arrow_right.x + arrow_right.y * arrow_right.y) / (arrow_left.x * arrow_left.x + arrow_left.y * arrow_left.y));
            return std::make_pair(angle, scale);
        }
        return std::make_pair(0.f, 1.f);
    };
    const auto num_tests = 5000;
    std::vector<float> scale_list(num_tests);
    for(size_t i = 0; i < num_tests; i++) {
        auto angle_scale = get_angle_scale();
        const float angle = angle_scale.first;
        const float scale = angle_scale.second;
        auto bucket_idx = size_t(std::round(angle * (num_buckets / float(M_PI * 2))));
        buckets[bucket_idx < num_buckets ? bucket_idx : 0]++;
        scale_list[i] = scale;
    }
    std::vector<float> buckets_padded(num_buckets + 6);
    std::copy_n(buckets.begin(), num_buckets, buckets_padded.begin() + 3);
    for(int i = 0; i < 3; i++) {
        buckets_padded[i] = buckets_padded[num_buckets + i];
        buckets_padded[3 + num_buckets + i] = buckets_padded[3 + i];
    }
    std::vector<float> histogram(num_buckets);
    for(size_t i = 0; i < num_buckets; i++){
        histogram[i] =
                buckets_padded[3 + i] * 0.383f
                + (buckets_padded[4 + i] + buckets_padded[2 + i]) * 0.242f
                + (buckets_padded[5 + i] + buckets_padded[1 + i]) * 0.061f
                + (buckets_padded[6 + i] + buckets_padded[i]) * 0.006f;
    }
    const long idx_max = std::max_element(histogram.begin(), histogram.end()) - histogram.begin();
    const auto angle = float(M_PI * 2) / num_buckets * idx_max;
    std::nth_element(scale_list.begin(), scale_list.begin() + num_tests / 2, scale_list.end());
    const float scale = scale_list[num_tests/2];//@fixme: this does not work
    return std::make_pair(angle, scale);
}

//@fixme: make it also scale-invariant
robust_gms_matcher_t::robust_gms_matcher_t(const std::vector<cv::KeyPoint> &vkp1, const cv::Size size1, const std::vector<cv::KeyPoint> &vkp2, const cv::Size size2, const std::vector<cv::DMatch> &vDMatches, bool pre_rotation)
{
    if(pre_rotation) {
        const float angle = find_rotation_scale(vkp1, vkp2, vDMatches, size1.width / 16, 64).first;
        typedef std::complex<float> complex;
        const complex rotation = {std::cos(angle), std::sin(angle)};
        complex corners[4] = {
                {0,           0},
                {float(size1.width), 0},
                {0,           float(size1.height)},
                {float(size1.width), float(size1.height)}};
        for (auto &c : corners)
            c = rotation * c;
        float x_min = INFINITY, y_min = INFINITY, x_max = -INFINITY, y_max = -INFINITY;
        for (const auto &c : corners) {
            x_min = std::min(x_min, c.real());
            x_max = std::max(x_max, c.real());
            y_min = std::min(y_min, c.imag());
            y_max = std::max(y_max, c.imag());
        }
        const complex offset = {-x_min, -y_min};
        const cv::Size size1_new{int(x_max - x_min + 1.f), int(y_max - y_min + 1.f)};
        assert(size1_new.width >= size1.width && size1_new.height > size1.height);

        std::vector<cv::KeyPoint> vkp1_new = vkp1;
        for (auto &p : vkp1_new) {
            const complex p_new = rotation * complex{p.pt.x, p.pt.y} + offset;
            p.pt = cv::Point2f{p_new.real(), p_new.imag()};
        }

        const cv::Size size2_new{
                int(std::round(size2.width * size1_new.width / float(size1.width))),
                int(std::round(size2.height * size1_new.height / float(size1.height)))
        };

        _matcher = std::make_unique<gms_matcher>(vkp1_new, size1_new, vkp2, size2_new, vDMatches);
    }
    else{
        _matcher = std::make_unique<gms_matcher>(vkp1, size1, vkp2, size2, vDMatches);
    }
}

robust_gms_matcher_t::~robust_gms_matcher_t(){}

int robust_gms_matcher_t::GetInlierMask(std::vector<bool> &vbInliers, bool WithScale, bool WithRotation)
{
    return _matcher->GetInlierMask(vbInliers, WithScale, WithRotation);
}
