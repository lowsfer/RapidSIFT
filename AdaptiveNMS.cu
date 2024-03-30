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
// Created by yao on 10/28/19.
//
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>
#include "KArray.h"
#include "utils_host.h"
#include "utils_sift.h"
#include "kernels.h"

namespace {
constexpr uint32_t ctaSize = 128;
struct alignas(8) TileDims {
    uint32_t m;
    uint32_t n;
};
constexpr TileDims thrdTileDims{8u, 4u};
constexpr uint32_t thrdNbTiles = 64u;
constexpr TileDims ctaTileDims{thrdTileDims.m * ctaSize, thrdTileDims.n * thrdNbTiles};

constexpr float getSqrDistance(float2 a, float2 b) {
    const float x = a.x - b.x;
    const float y = a.y - b.y;
    return x * x + y * y;
}

__global__ void kernel_findSuppressionRadii __launch_bounds__ (ctaSize)(float *__restrict__ const minSqrDistance,
                            const float2 *__restrict__ const locations, const float *__restrict__ const responses,
                            const uint32_t nbKPoints,
                            const float robustCoeff = 1.11f) {
    assert(robustCoeff > 1.f);
    if (ctaTileDims.n * blockIdx.x >= ctaTileDims.m * blockIdx.y + ctaTileDims.m - 1) {
        return;
    }
    const auto idxThrdCta = threadIdx.x;
    struct PtData {
        float2 loc;
        float respThres;
        float minSqrDist;
    };
    KArray<PtData, thrdTileDims.m> rows{};
    for (uint32_t i = 0; i < thrdTileDims.m; i++) {
        auto &r = rows[i];
        r.minSqrDist = INFINITY;
        const uint32_t idxKPoint = ctaTileDims.m * blockIdx.y + ctaSize * i + idxThrdCta;
        if (idxKPoint < nbKPoints) {
            r.loc = locations[idxKPoint];
            r.respThres = responses[idxKPoint] * robustCoeff;
        }
    }
    __shared__ KArray<float2, thrdNbTiles, thrdTileDims.n> ctaLocations;
    __shared__ KArray<float, thrdNbTiles, thrdTileDims.n> ctaResponses;
    {
        const auto ldIters = ctaTileDims.n / ctaSize;
        KArray<float2, ldIters> locRegBuf;
        KArray<float, ldIters> respRegBuf;
        for (uint32_t iter = 0; iter < ldIters; iter++) {
            locRegBuf[iter] = float2{INFINITY, INFINITY};
            respRegBuf[iter] = 0.f;
            const uint32_t idxKPoint = ctaTileDims.n * blockIdx.x + ctaSize * iter + idxThrdCta;
            if (idxKPoint < nbKPoints) {
                locRegBuf[iter] = locations[idxKPoint];
                respRegBuf[iter] = responses[idxKPoint];
            }
        }
        for (uint32_t iter = 0; iter < ldIters; iter++) {
            ctaLocations[0].data[ctaSize * iter + idxThrdCta] = locRegBuf[iter];
            ctaResponses[0].data[ctaSize * iter + idxThrdCta] = respRegBuf[iter];
        }
    }
    __syncthreads();
    constexpr uint32_t nbRegBufs = 2; // double buffer;
    KArray<float2, nbRegBufs, thrdTileDims.n> thrdLocRegBufs;
    KArray<float, nbRegBufs, thrdTileDims.n> thrdRespRegBufs;
    auto ldThrdLocResp = [&](uint32_t idxTile, uint32_t idxBuf) {
        assert(idxBuf == idxTile % nbRegBufs);
        thrdLocRegBufs[idxBuf] = ctaLocations[idxTile];
        thrdRespRegBufs[idxBuf] = ctaResponses[idxTile];
    };
    auto updateMinSqrDist = [&](uint32_t idxTile, uint32_t idxBuf) {
        assert(idxBuf == idxTile % nbRegBufs);
        const KArray<float2, thrdTileDims.n> &thrdLoc = thrdLocRegBufs[idxBuf];
        const KArray<float, thrdTileDims.n> &thrdResp = thrdRespRegBufs[idxBuf];
        for (uint32_t j = 0; j < thrdTileDims.n; j++) {
            for (uint32_t i = 0; i < thrdTileDims.m; i++)
                if (thrdResp[j] > rows[i].respThres) {
                    const auto idxCol = ctaTileDims.n * blockIdx.x + thrdTileDims.n * idxTile + j;
                    const auto idxRow = ctaTileDims.m * blockIdx.y + ctaSize * i + idxThrdCta;
                    assert(idxCol < idxRow);
                    (void) idxCol;
                    (void) idxRow;
                    (void) idxTile;
                    const float sqrDist = getSqrDistance(rows[i].loc, thrdLoc[j]);
                    if (sqrDist < rows[i].minSqrDist) {
                        rows[i].minSqrDist = sqrDist;
                    }
                }
        }
    };
    ldThrdLocResp(0u, 0u);
#pragma unroll(1)
    for (uint32_t i = 0; i < (thrdNbTiles - 1u) / nbRegBufs; i++) {
#pragma unroll
        for (uint32_t j = 0; j < nbRegBufs; j++) {
            const uint32_t idxTile = nbRegBufs * i + j;
            ldThrdLocResp(idxTile + 1, (j + 1) % nbRegBufs); // prefetch
            updateMinSqrDist(idxTile, j % nbRegBufs);
        }
    }
#pragma unroll
    for (uint32_t idxTile = (thrdNbTiles - 1u) / nbRegBufs * nbRegBufs; idxTile < thrdNbTiles; idxTile++) {
        if (idxTile < thrdNbTiles - 1) {
            ldThrdLocResp(idxTile + 1, (idxTile + 1) % nbRegBufs); // prefetch
        }
        updateMinSqrDist(idxTile, idxTile % nbRegBufs);
    }
    for (uint32_t i = 0; i < thrdTileDims.m; i++) {
        const uint32_t idxKPoint = ctaTileDims.m * blockIdx.y + ctaSize * i + idxThrdCta;
        if (idxKPoint < nbKPoints) {
            using CmpType = int32_t;
            static_assert(sizeof(CmpType) == sizeof(minSqrDistance[0]));
            atomicMin(reinterpret_cast<CmpType*>(&minSqrDistance[idxKPoint]), reinterpret_cast<const CmpType&>(rows[i].minSqrDist));
        }
    }
}

template <typename T>
__global__ void kernelInit1D(T* __restrict__ const data, uint32_t size, T value){
    const uint32_t idxThrd = ctaSize * blockIdx.x + threadIdx.x;
    if (idxThrd < size){
        data[idxThrd] = value;
    }
}

}

cudaError_t cudaFindSuppressionRadii(float *__restrict__ const minSqrDistance,
                                      const float2 *__restrict__ const locations, const float *__restrict__ const responses,
                                      const uint32_t nbKPoints,
                                      const float robustCoeff /*= 1.11f*/,
                                      cudaStream_t stream){
    kernelInit1D<float><<<div_ceil(nbKPoints, ctaSize), ctaSize, 0, stream>>>(minSqrDistance, nbKPoints, INFINITY);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    const dim3 dimGrid{div_ceil(nbKPoints, ctaTileDims.n), div_ceil(nbKPoints, ctaTileDims.m)};
    kernel_findSuppressionRadii<<<dimGrid, ctaSize, 0, stream>>>(minSqrDistance, locations, responses, nbKPoints, robustCoeff);
    return cudaGetLastError();
}

