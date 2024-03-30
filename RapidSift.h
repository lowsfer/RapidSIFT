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

#pragma once
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>
#include "public_types.h"
#include <future>

class RapidSift {
public:
    virtual ~RapidSift() = default;

    virtual std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> detect_and_describe(
            std::function<const void*()>&& src_getter, int width, int height, float thres_contrast = 0.04f, bool up_sample = true) = 0;
    virtual std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> uniform_detect_and_compute(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio = 2.f,
            float init_thres_contrast = 0.04f, bool up_sample = true,
            float min_thres_contrast = 0.002f) = 0;
    // returns {kpoints, descriptors, mask for abstract samples}
    virtual std::future<std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>>> uniform_detect_compute_and_abstract(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio = 2.f,
            float init_thres_contrast = 0.04f, bool up_sample = true,
            float min_thres_contrast = 0.002f, uint32_t nbAbstractSamples = 300u) = 0;
    virtual size_t get_num_workers() const = 0;
};

//sorted by response: strong to weak
[[nodiscard]] std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>
sort_kpoints(const std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>& kpoints);

// Seems SOSNet is better for oblique and RootSIFT is better for downward planar datasets.
// Maybe because the SOSNet weights we use were trained on HPatch dataset.
extern "C" [[nodiscard]] RapidSift* create_sift(size_t num_works, DescType descType);
