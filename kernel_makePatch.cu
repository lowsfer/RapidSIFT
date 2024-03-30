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
// Created by yao on 20/01/18.
//

#ifdef __CLION_IDE__
#define __CUDACC__
#endif

#include "kernels.h"
#include "utils_host.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <limits>

namespace cg = cooperative_groups;

template<int num_threads>
__device__ __forceinline__ float thread_group_sum(const cg::thread_block_tile<num_threads>& g, const float val){
    static_assert(num_threads == 1 || num_threads == 2 || num_threads == 4 || num_threads == 8 || num_threads == 16 || num_threads == 32, "fatal error");
    float sum = val;
#pragma unroll
    for(int i = num_threads / 2; i != 0; i /= 2){
        const float sum_other = g.shfl_xor(sum, i);
        sum += sum_other;
    }
    return sum;
}

__device__ __forceinline__ uint32_t lane_id()
{
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;\n" : "=r"(laneid));
    return laneid;
}

inline __device__ int32_t pack_int8x4(int32_t (&a)[4]) {
    int32_t res;
    asm volatile(
        "{\n" \
        ".reg .u32 r4;\n" \
        "cvt.pack.sat.s8.s32.b32   r4, %4, %3,  0;\n" \
        "cvt.pack.sat.s8.s32.b32   %0, %2, %1, r4;\n" \
        "}" \
        : "=r"(res) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]));
    return res;
}
inline __device__ int32_t f2i(float a) {
    int32_t res;
    asm volatile("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(res) : "f"(a));
    return res;
}
inline __device__ uint32_t float4_to_s8x4( float4 in ) {
    uint32_t ret;
    int32_t tmp[4];
    tmp[0] = f2i( in.x );
    tmp[1] = f2i( in.y );
    tmp[2] = f2i( in.z );
    tmp[3] = f2i( in.w );
    ret = pack_int8x4( tmp );
    return ret;
}

static constexpr int thrds_per_pt = 32;
static constexpr int pts_per_cta = 4;
static constexpr int block_dim = thrds_per_pt * pts_per_cta;

__global__ void kernel_makePatch(
        cudaTextureObject_t const img, // the original image, using normalized float coordinates
        float2 const rcpImgSize,
        const KeyPoint* __restrict__ const kpoints,
        const uint32_t num_kpoints,//min(max_num_kpoints, kpoints->count)
        float const magFactor,
        float const rcpOutQuantScale,
        Patch32x32* __restrict__ const patches)
{
    const cg::thread_block cta = cg::this_thread_block();
    const cg::thread_block_tile<thrds_per_pt> g = cg::tiled_partition<thrds_per_pt>(cta);
    const int idx_grp = threadIdx.x / g.size();
    const int idx = pts_per_cta * blockIdx.x + idx_grp;
    if(idx < num_kpoints) {
        const KeyPoint kpt = kpoints[idx];
        const float angle = kpt.angle * (2 * float(M_PI) / 256);
        const float2 location = kpt.location;
        const float size = __half2float(kpt.size);

        float patch[Patch32x32::patchSize];

        float const s = magFactor * size / Patch32x32::patchSize;
        float sin_t, cos_t;
        sincos(angle, &sin_t, &cos_t);
        sin_t *= s;
        cos_t *= s;
        int const lane = lane_id();
        float2 const base = {
            location.x - 15.5f * (cos_t - sin_t), // @fixme: the original SoSNet code uses 16
            location.y - 15.5f * (sin_t + cos_t)
        };

        constexpr int vecSize = 16;
        constexpr int nbIters = 32*32/32/vecSize;
        #pragma unroll
        for (int n = 0; n < nbIters; n++) {
            int const linearBase = vecSize * 32 * n + vecSize * lane;
            int const i = linearBase / 32;
            int const jBase = linearBase % 32;
            #pragma unroll
            for (int e = 0; e < vecSize; e++) {
                int const j = jBase + e;
                float const x = base.x + cos_t * j - sin_t * i;
                float const y = base.y + sin_t * j + cos_t * i;
                patch[vecSize * n + e] = tex2D<float>(img, x * rcpImgSize.x, y * rcpImgSize.y);
            }
        }

        auto computeWarpSum = [&](bool squared) {
            float sum = 0;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                sum += (squared ? sqr(patch[i]) : patch[i]);
            }
            #pragma unroll
            for (uint32_t i = 16; i != 0; i /= 2) {
                sum += __shfl_xor_sync(~0U, sum, i, 32);
            }
            return sum;
        };
        float const mean = computeWarpSum(false) / 1024;
        #pragma unroll
        for (float& v : patch) {
            v -= mean;
        }
        float const rcpSigma = rsqrtf(std::max(computeWarpSum(true) / 1024, 1E-20f));
        float const factor = rcpSigma * rcpOutQuantScale;
        #pragma unroll
        for (float& v : patch) {
            v *= factor;
        }
        #pragma unroll
        for (int n = 0; n < nbIters; n++) {
            alignas(16) uint32_t out[vecSize / 4];
            #pragma unroll
            for (int i = 0; i < vecSize / 4; i++) {
                out[i] = float4_to_s8x4(reinterpret_cast<float4 const&>(patch[vecSize*n + 4*i]));
            }
            int const linearBase = vecSize * 32 * n + vecSize * lane;
            static_assert(sizeof(float4) == sizeof(out));
            reinterpret_cast<float4&>(patches[idx].data[0][linearBase]) = reinterpret_cast<float4 const&>(out);
        }
#if 0
        __syncwarp();
        const float2 refPt = {275.845581, 677.882324};
        if (sqr(location.x - refPt.x) + sqr(location.y - refPt.y) < sqr(3) && lane == 0)
        {
            for (int i = 0; i < 1024; i++) {
                printf("%d, ", patches[idx].data[0][i]);
            }
            printf("\n");
        }
#endif
    }
}

cudaError_t cuda_makePatch(
        cudaTextureObject_t const img, // the orignal image
        int2 imgSize, // the original image size
        const GPUArray<KeyPoint>* __restrict__ kpoints,
        const uint32_t num_kpoints,//min(max_num_kpoints, kpoints->count)
        float const magFactor,
        float const rcpOutQuantScale,
        Patch32x32* const patches,
        const cudaStream_t stream)
{
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    if(num_kpoints != 0) {
        const int grid_dim = div_ceil((int) num_kpoints, pts_per_cta);
        kernel_makePatch<<<grid_dim, block_dim, 0, stream>>>(img, float2{1.f / imgSize.x, 1.f / imgSize.y}, kpoints->data, num_kpoints, magFactor, rcpOutQuantScale, patches);
    }
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    return cudaGetLastError();
}
