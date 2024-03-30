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
// Created by yao on 21/01/18.
//

#pragma once

#include "RapidSift.h"
#include "SIFT_worker.h"

class sift_master : public RapidSift {
public:
    explicit sift_master(size_t num_workers, DescType DescType = DescType::kSIFT);

    ~sift_master() override;

    std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> detect_and_describe(
            std::function<const void*()>&& src_getter, int width, int height, float contrast_threshold = 0.04f, bool up_sample = true) override ;
    std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> uniform_detect_and_compute(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
            float init_thres_contrast/* = 0.04f*/, bool up_sample/* = true*/,
            float min_thres_contrast) override;
    std::future<std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>>>
    uniform_detect_compute_and_abstract(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
            float init_thres_contrast/* = 0.04f*/, bool /*up_sample = true*/,
            float min_thres_contrast,
            uint32_t nbAbstractSamples/* = 300u*/) override;
    size_t get_num_workers() const override;

private:
    std::vector<std::unique_ptr<SIFT_worker>> _workers;
    uint32_t idx_next = 0;
};
