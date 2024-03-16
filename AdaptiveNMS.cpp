//
// Created by yao on 10/29/19.
//

#include <stdint-gcc.h>
#include "AdaptiveNMS.h"
#include "fp16.h"

cudaError_t cudaFindSuppressionRadii(float *__restrict__ minSqrDistance,
                                     const float2 *__restrict__ locations,
                                     const float *__restrict__ responses,
                                     uint32_t nbKPoints,
                                     float robustCoeff /*= 1.11f*/,
                                     cudaStream_t stream);

// note that we may return a bit less than targetNbKPoints in rare cases
float cudaGetDecisionRadius(
        const float * minSqrDistance, uint32_t nbKPoints, uint32_t targetNbKPoints){
    assert(targetNbKPoints <= nbKPoints);
    std::vector<float> buffer(minSqrDistance, minSqrDistance + nbKPoints);

    assert(targetNbKPoints <= nbKPoints);
    std::nth_element(buffer.begin(), buffer.begin() + targetNbKPoints, buffer.begin() + nbKPoints, std::greater<>{});
    return buffer[targetNbKPoints];
}


uint32_t AdaptiveNMS::filterDevKPointsSync(GPUArray<KeyPoint> *devKPoints, uint32_t targetNbKPoints, float robustCoeff,
                                           bool sortResultBySize) {
    mDevKPoints = devKPoints;
    const std::vector<bool> mask = uniformSample(devKPoints, targetNbKPoints, robustCoeff);
    require(mask.size() == *mNbKPoints);

    const std::vector<KeyPoint> allKPoints(mHostKPoints.begin(), mHostKPoints.begin() + *mNbKPoints);
    mHostKPoints.clear();
    for (uint32_t i = 0; i < *mNbKPoints; i++){
        if (mask.at(i)){
            mHostKPoints.push_back(allKPoints.at(i));
        }
    }
    //@fixme: see if this sorting improves GPU performance - may make it a bit more cache-friendly.
    if (sortResultBySize){
        std::sort(mHostKPoints.begin(), mHostKPoints.end(), [](const auto& a, const auto& b){return a.size < b.size;});
    }
    *mFilteredNbKPoints = mHostKPoints.size();
    checkCudaError(cudaMemcpyAsync(&mDevKPoints->count, mFilteredNbKPoints.get(), sizeof(uint32_t), cudaMemcpyDeviceToHost, mStream));
    checkCudaError(cudaMemcpyAsync(mDevKPoints->data, mHostKPoints.data(), sizeof(KeyPoint) * mHostKPoints.size(), cudaMemcpyDeviceToHost, mStream));
    return *mFilteredNbKPoints;
}


std::vector<bool> AdaptiveNMS::uniformSample(const GPUArray<KeyPoint> *devKPoints, uint32_t targetNbKPoints,
        float robustCoeff) {
    checkCudaError(cudaMemcpyAsync(mNbKPoints.get(), &devKPoints->count, sizeof(devKPoints->count), cudaMemcpyDeviceToHost, mStream));
    syncStream();
    if (targetNbKPoints >= *mNbKPoints){
        return std::vector<bool>(*mNbKPoints, true);
    }
    reserve(*mNbKPoints);
    checkCudaError(cudaMemcpyAsync(mHostKPoints.data(), devKPoints->data, sizeof(KeyPoint) * *mNbKPoints, cudaMemcpyDeviceToHost, mStream));
    syncStream();
    std::sort(mHostKPoints.begin(), mHostKPoints.end(), [](const KeyPoint& a, const KeyPoint& b){
        return half2float(a.response) > half2float(b.response);
    });

    for (uint32_t i = 0; i < *mNbKPoints; i++){
        mHostLocation.at(i) = mHostKPoints.at(i).location;
        mHostResponse.at(i) = half2float(mHostKPoints.at(i).response);
    }
    checkCudaError(cudaMemcpyAsync(mDevLocation.get(), mHostLocation.data(), sizeof(float2) * *mNbKPoints, cudaMemcpyHostToDevice, mStream));
    checkCudaError(cudaMemcpyAsync(mDevResponse.get(), mHostResponse.data(), sizeof(float) * *mNbKPoints, cudaMemcpyHostToDevice, mStream));
    checkCudaError(cudaFindSuppressionRadii(mDevMinSqrDistance.get(), mDevLocation.get(), mDevResponse.get(), *mNbKPoints, robustCoeff, mStream));
    checkCudaError(cudaMemcpyAsync(mHostMinSqrDistance.data(), mDevMinSqrDistance.get(), sizeof(float) * *mNbKPoints, cudaMemcpyDeviceToHost, mStream));
    syncStream();
    const float decisionRadius = cudaGetDecisionRadius(mHostMinSqrDistance.data(), *mNbKPoints, targetNbKPoints);

    std::vector<bool> mask(*mNbKPoints, false);
    for (uint32_t i = 0; i < *mNbKPoints; i++){
        if (mHostMinSqrDistance.at(i) > decisionRadius){
            mask.at(i) = true;
        }
    }
    return mask;
}

