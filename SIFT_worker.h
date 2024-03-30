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
// Created by yao on 13/01/18.
//

#pragma once

#include <future>
#include "Octave.h"
#include "cuda_async_copy_engine.h"
#include "kernels.h"
#include "ThreadPool.h"
#include "AdaptiveNMS.h"
#if RAPIDSIFT_ENABLE_SOSNET
#include <SOSNet.h>
#endif
class SIFT_worker {
public:
    SIFT_worker(DescType descType);

    ~SIFT_worker();

    //async API
    std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> detect_and_compute_async(
            std::function<const void*()>&& src_getter, int width, int height, float thres_contrast/* = 0.04f*/, bool up_sample/* = true*/);

    std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>> uniform_detect_and_compute_async(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
            float init_thres_contrast/* = 0.04f*/, bool up_sample/* = true*/,
            float min_thres_contrast);

    std::future<std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>>>
    uniform_detect_compute_and_abstract_async(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
            float init_thres_contrast/* = 0.04f*/, bool up_sample/* = true*/,
            float min_thres_contrast, uint32_t nbAbstractSamples/* = 300u*/);

private:
    //synchronous API (not thread-safe)
    void feed(std::function<const void*()>&& src_getter, int width, int height, float thres_contrast, bool up_sample);
    void feed(std::function<const void*()>&& src_getter, int width, int height,
            uint32_t targetNbKPoints, float minOverDetectRatio, float init_thres_contrast, bool up_sample,
            float min_thres_contrast);
    std::vector<KeyPoint> get_keypoints();
    std::vector<SiftDescriptor> get_descriptors();

private:
    void preprocess(std::function<const void*()>&& src_getter, int width, int height, bool up_sample = true);

    void build_DoG_layers(bool up_sample);

    void find_keypoints(float thres_contrast);

    void assign_orientation(bool up_sample);

    void describe(bool up_sample);

    void reserve_extremas(uint32_t max_num_extremas);

    void reserve_kpoints(uint32_t max_num_kpoints);

private:
    std::mutex _mutex;
    std::unique_ptr<CUstream_st, cuda_stream_deleter> _stream = [](){return std::unique_ptr<CUstream_st, cuda_stream_deleter>(make_cuda_stream());}();
    cuda_async_copy_engine _copy_engine = cuda_async_copy_engine{_stream.get()};
    Pitched2DTexture<uint8_t> _dev_input;//@fixme: replace with an cudaArray-based texture

    float _sigma = 1.6f;
//    float _thres_contrast = 0.04f; // changed to function argument
    float _thres_edge = 10.f;

    int2 _img_size;
    size_t _num_octave_layers = 3; // Lowe's paper and OpenCV uses 3, but 48 seems to be best

    size_t _num_octaves = 0;
    std::vector<Octave> _octaves;//do not use _octaves.size(), use _num_octaves instead

    std::array<Pitched2DTexture<float>, 2> _workspace;

    std::unique_ptr<cudaTextureObject_t[], cuda_device_deleter> _textures;//allocated on device
    size_t _capacity_textures = 0;
    std::unique_ptr<GPUArray<ScaleSpaceExtrema>, cuda_device_deleter> _extremas;
    uint32_t _max_num_extremas = 0;
    std::unique_ptr<GPUArray<KeyPoint>, cuda_device_deleter> _kpoints;
    uint32_t _max_num_kpoints = 0;
    std::unique_ptr<SiftDescriptor[], cuda_device_deleter> _descriptors;
#if RAPIDSIFT_ENABLE_SOSNET
    std::unique_ptr<Patch32x32[], cuda_device_deleter> _patches;// for SOSNet
    std::unique_ptr<ISOSNetInfer> _sosnet;
#endif
    uint32_t _num_kpoints = 0;//set by get_kpoints() and used by get_descriptors(). Should only be used in cuda callbacks in stream

    float _predicated_thres_contrast {0.04f};

    DescType mDescType = DescType::kSIFT;

    AdaptiveNMS _anms_filter{_stream.get()};

    ThreadPool _thread_pool;

    void syncStream() const {
        checkCudaError(cudaStreamSynchronize(_stream.get()));
    }
};


