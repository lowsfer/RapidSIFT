//
// Created by yao on 1/05/18.
//

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

struct alignas(16) KeyPoint{
    float2 location;// texture coordinates. (0,0) is at the upperleft corner, NOT the centre of the upperleft pixel. UpperLeft pixel is (0.5, 0.5)
    half size;
    half response;
    uint8_t angle;//fixed-point, value = (angle / 256.f) * 2 * M_PI
    int8_t octave;//can be -1 if up sampled
    uint8_t layer;//idx of DoG layer in the octave.
    int8_t layer_offset;//[-128, 127] mapped to [-(0.5-1/512), 0.5-1/512]; layer + (layer_offset + 0.5f)/256 falls in range [0, nbOctaveLayers+1], where nbOctaveLayers is 3.
};
static_assert(sizeof(KeyPoint) == 16, "fatal error");

// default width of descriptor histogram array
static const int sift_desc_width = 4;
// default number of bins per histogram in descriptor array
static const int sift_desc_hist_bins = 8;

struct alignas(128) SiftDescriptor{
    uint8_t data[sift_desc_width][sift_desc_width][sift_desc_hist_bins];
};
static_assert(sizeof(SiftDescriptor) == 128, "fatal error");

enum class DescType : int32_t {
    kSIFT,
    kRootSIFT,
    kSOSNet
};
