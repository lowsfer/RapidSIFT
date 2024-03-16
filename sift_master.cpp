//
// Created by yao on 21/01/18.
//

#include "sift_master.h"
#include "fp16.h"

sift_master::sift_master(size_t num_workers, DescType descType) {
    while(_workers.size() < num_workers){
        _workers.emplace_back(std::make_unique<SIFT_worker>(descType));
    }
}

sift_master::~sift_master() {
}

std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>>
sift_master::detect_and_describe(std::function<const void *()> &&src_getter, int width, int height, float thres_contrast, bool up_sample) {
    const auto& worker = _workers[idx_next];
    idx_next = (idx_next + 1) % uint32_t(_workers.size());
    return worker->detect_and_compute_async(std::move(src_getter), width, height, thres_contrast, up_sample);
}

std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> sift_master::uniform_detect_and_compute(
        std::function<const void*()>&& src_getter, int width, int height,
        // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
        uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
        float init_thres_contrast/* = 0.04f*/, bool up_sample/* = true*/, float min_thres_contrast){
    const auto& worker = _workers[idx_next];
    idx_next = (idx_next + 1) % uint32_t(_workers.size());
    return worker->uniform_detect_and_compute_async(std::move(src_getter), width, height, targetNbKPoints, minOverDetectRatio, init_thres_contrast, up_sample, min_thres_contrast);
}

std::future<std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>>>
    sift_master::uniform_detect_compute_and_abstract(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
            float init_thres_contrast/* = 0.04f*/, bool up_sample/* = true*/,
            float min_thres_contrast,
            uint32_t nbAbstractSamples/* = 300u*/)
{
    const auto& worker = _workers[idx_next];
    idx_next = (idx_next + 1) % uint32_t(_workers.size());
    return worker->uniform_detect_compute_and_abstract_async(std::move(src_getter), width, height, targetNbKPoints, minOverDetectRatio, init_thres_contrast, up_sample, min_thres_contrast, nbAbstractSamples);
}

size_t sift_master::get_num_workers() const{
    return _workers.size();
}