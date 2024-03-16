//
// Created by yao on 10/29/19.
//

#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "public_types.h"
#include "kernels.h"
#include "utils_host.h"

class AdaptiveNMS
{
public:
    explicit AdaptiveNMS(cudaStream_t stream) : mStream{stream}{}
    uint32_t filterDevKPointsSync(GPUArray<KeyPoint> *devKPoints, uint32_t targetNbKPoints, float robustCoeff = 1.11f,
                                  bool sortResultBySize = false);
    std::vector<bool> uniformSample(const GPUArray<KeyPoint> *devKPoints, uint32_t targetNbKPoints,
        float robustCoeff = 1.11f);

private:
    void reserve(uint32_t size){
        mHostKPoints.resize(size);
        mHostLocation.resize(size);
        mHostResponse.resize(size);
        mHostMinSqrDistance.resize(size);
        if (size > mDevCapacity){
            mDevLocation.reset((float2*)device_alloc(sizeof(*mDevLocation) * size));
            mDevResponse.reset((float*)device_alloc(sizeof(*mDevResponse) * size));
            mDevMinSqrDistance.reset((float*)device_alloc(sizeof(*mDevMinSqrDistance) * size));
            mDevCapacity = size;
        }
    }
private:
    GPUArray<KeyPoint>* mDevKPoints = nullptr;
    cudaStream_t mStream = nullptr;
    std::unique_ptr<uint32_t, cuda_host_deleter> mNbKPoints{static_cast<uint32_t*>(host_alloc(sizeof(uint32_t)))};
    std::vector<KeyPoint, cuda_host_allocator<KeyPoint>> mHostKPoints;

    std::vector<float2, cuda_host_allocator<float2>> mHostLocation;
    std::vector<float, cuda_host_allocator<float>> mHostResponse;
    std::vector<float, cuda_host_allocator<float>> mHostMinSqrDistance;
    std::unique_ptr<float2, cuda_device_deleter> mDevLocation;
    std::unique_ptr<float, cuda_device_deleter> mDevResponse;
    std::unique_ptr<float, cuda_device_deleter> mDevMinSqrDistance;
    uint32_t mDevCapacity{0};

    std::unique_ptr<uint32_t, cuda_host_deleter> mFilteredNbKPoints{static_cast<uint32_t*>(host_alloc(sizeof(uint32_t)))};

    void syncStream() const {
        checkCudaError(cudaStreamSynchronize(mStream));
    }
};