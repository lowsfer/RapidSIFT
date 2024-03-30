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
// Created by yao on 10/29/19.
//

#include "../AdaptiveNMS.h"
#include "../fp16.h"
#include <boost/test/unit_test.hpp>
#include <cstdint>
#include <opencv2/core.hpp>

void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                   const uint32_t numToKeep );

const cv::KeyPoint KeyPoint2cvKeyPoint(const KeyPoint& kpt);

BOOST_AUTO_TEST_CASE(AdaptiveNMSTest)
{
    const uint32_t nbKPoints = 2000;
    const uint32_t targetNbKPoints = 200;
    const float robustCoeff = 1.11f;
    std::mt19937_64 rng{};
    std::uniform_real_distribution<float> distResp{};
    std::normal_distribution<float> distLoc{500.f, 300.f};

    const uint32_t bufBytes = sizeof(GPUArray<KeyPoint>) + sizeof(KeyPoint) * nbKPoints;
    std::unique_ptr<GPUArray<KeyPoint>, cuda_device_deleter> kpoints{(GPUArray<KeyPoint>*)managed_alloc(bufBytes)};
    kpoints->count = nbKPoints;
    for (uint32_t i = 0; i < nbKPoints; i++){
        kpoints->data[i].location = {distLoc(rng), distLoc(rng)};
        kpoints->data[i].response = float2half<half>(distResp(rng));
    }

    std::vector<cv::KeyPoint> refKPoints(nbKPoints);
    std::transform(kpoints->data, kpoints->data + nbKPoints, refKPoints.begin(), KeyPoint2cvKeyPoint);

    const cudaStream_t stream = nullptr;
    AdaptiveNMS anms(stream);
    int deviceId = 0;
    checkCudaError(cudaGetDevice(&deviceId));
    checkCudaError(cudaMemPrefetchAsync(kpoints.get(), bufBytes, deviceId, stream));
    anms.filterDevKPointsSync(kpoints.get(), targetNbKPoints, robustCoeff, false);
//    std::sort(kpoints->data, &kpoints->data[kpoints->count], [](auto a, auto b){return half2float(a.response) > half2float(b.response);});


    adaptiveNonMaximalSuppresion(refKPoints, targetNbKPoints);

    BOOST_CHECK_EQUAL(kpoints->count, refKPoints.size());
    for (uint32_t i = 0; i < kpoints->count; i++){
        BOOST_CHECK_EQUAL(kpoints->data[i].location.x, refKPoints[i].pt.x);
        BOOST_CHECK_EQUAL(kpoints->data[i].location.y, refKPoints[i].pt.y);
    }
}

void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
                                   const uint32_t numToKeep )
{
    if(keypoints.size() < numToKeep) { return; }

    //
    // Sort by response
    //
    std::sort( keypoints.begin(), keypoints.end(),
               [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
               {
                   return lhs.response > rhs.response;
               } );

    std::vector<cv::KeyPoint> anmsPts;

    std::vector<double> radii;
    radii.resize( keypoints.size() );
    std::vector<double> radiiSorted;
    radiiSorted.resize( keypoints.size() );

    const float robustCoeff = 1.11; // see paper

    //@fixme: Use a grid to accelerate.
    for(uint32_t i = 0; i < keypoints.size(); ++i )
    {
        const float response = keypoints[i].response * robustCoeff;
        double radius = std::numeric_limits<double>::max();
        for( uint32_t j = 0; j < i && keypoints[j].response > response; ++j )
        {
            radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
        }
        radii[i]       = radius;
        radiiSorted[i] = radius;
    }

    std::sort( radiiSorted.begin(), radiiSorted.end(),
               [&]( const double& lhs, const double& rhs )
               {
                   return lhs > rhs;
               } );

    const double decisionRadius = radiiSorted[numToKeep];
    for (uint32_t i = 0; i < radii.size(); ++i)
    {
        if (radii[i] > decisionRadius)
        {
            anmsPts.push_back(keypoints[i]);
        }
    }

    anmsPts.swap(keypoints);
}

const cv::KeyPoint KeyPoint2cvKeyPoint(const KeyPoint& kpt){
    cv::KeyPoint ret;
    ret.pt = cv::Point2f(kpt.location.x, kpt.location.y);
    ret.size = half2float(kpt.size);
    ret.angle = kpt.angle * 360.f / 256.f;
    ret.octave = kpt.octave;
    ret.response = half2float(kpt.response);
    ret.class_id = -1;
    return ret;
}
