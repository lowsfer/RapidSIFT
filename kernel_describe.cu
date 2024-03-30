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

static constexpr int thrds_per_pt = SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH;
static constexpr int pts_per_cta = 8;
static constexpr int block_dim = thrds_per_pt * pts_per_cta;

template <bool useRootSIFT>
__global__ void kernel_describe(
        const cudaTextureObject_t* __restrict__ const gauss_layers,
        const int num_octave_layers,//layers per octave.
        const int num_octaves,
        float2 const octave0_size, // up-scaled size if up-scaling is used.
        const bool up_sampled,
        const KeyPoint* __restrict__ const kpoints,
        const uint32_t num_kpoints,//min(max_num_kpoints, kpoints->count)
        const float max_radius,
        SiftDescriptor* const descriptors)
{
    const cg::thread_block cta = cg::this_thread_block();
    const cg::thread_block_tile<thrds_per_pt> g = cg::tiled_partition<thrds_per_pt>(cta);
    const int idx_grp = threadIdx.x / g.size();
    const int idx = pts_per_cta * blockIdx.x + idx_grp;
    if(idx < num_kpoints) {
        const KeyPoint kpt = kpoints[idx];
        const int octave = up_sampled ? kpt.octave + 1 : kpt.octave;
        float2 const layerSize = {octave0_size.x / (1<<octave), octave0_size.y / (1<<octave)};
#if 1
        const float layerDelta = (kpt.layer_offset + 0.5f) * (1.f / 256);
        int32_t const idxGaussLayerInOctave = (int)floor(kpt.layer + layerDelta);
        assert(in_range(idxGaussLayerInOctave, 0, num_octave_layers + 1));
        const cudaTextureObject_t layer = gauss_layers[(num_octave_layers + 1) * octave + idxGaussLayerInOctave];
#else
        const cudaTextureObject_t layer = gauss_layers[num_octave_layers * octave + kpt.layer - 1];
#endif
        const float angle = kpt.angle * (2 * float(M_PI) / 256);
        // const float angle = isDebug ? 0.f : kpt.angle * (2 * float(M_PI) / 256);
        float const locSizeScale = (up_sampled ? 2.f : 1.f) / (1 << octave);
        const float2 location = {//in texture coordinates
                kpt.location.x * locSizeScale,
                kpt.location.y * locSizeScale
        };
        const float size = __half2float(kpt.size) * locSizeScale;
        const float scale = size * 0.5f;
        const float exp_scale = -2.f / sqr(SIFT_DESCR_WIDTH);
        const float hist_width = SIFT_DESCR_SCL_FCTR * scale;
        const float radius = std::min(max_radius, std::roundf(hist_width * 1.4142f * (SIFT_DESCR_WIDTH + 1) * 0.5f));
//        radius = std::min(radius, (int) sqrt(((double) img.cols)*img.cols + ((double) img.rows)*img.rows));
        //@fixme: using fixed-point because shared memory do not support native float atomicAdd. Change when GPU has smem float atomicAdd in the future
        __shared__ uint32_t hist_cta[pts_per_cta][SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS];//[0, 65536] maps to [0, 1.f]
        auto float2fixed = [](float val){return uint32_t(roundf(val * float(1<<16)));};
        auto fixed2float = [](uint32_t val){return val * (1.f / (1<<16));};

        uint32_t (&hist_grp)[SIFT_DESCR_WIDTH][SIFT_DESCR_WIDTH][SIFT_DESCR_HIST_BINS] = hist_cta[idx_grp];
        constexpr int elems_per_thrd = div_ceil(SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS, thrds_per_pt);
        for(int i = 0; i < elems_per_thrd; i++) {
            const int k = i * g.size() + g.thread_rank();
            if(k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS) {
#if 1
                hist_grp[0][g.thread_rank()][i] = 0;//@fixme:this has more bank conflict but turn out to be faster, why???
#else
                hist_grp[0][0][k] = 0;
#endif
            }
        }
        g.sync();
        for (int i = 0; i < sizeof(hist_grp) / sizeof(hist_grp[0][0][0]); i++) {
            assert(hist_grp[0][0][i] == 0);
        }

        constexpr int d = SIFT_DESCR_WIDTH;
        constexpr int n = SIFT_DESCR_HIST_BINS;

        float cos_t, sin_t;
        {
            sincos(-angle, &sin_t, &cos_t);
            const float tmp = 1.f / hist_width;
            cos_t *= tmp; sin_t *= tmp;
        }
        auto startOffset = [] (float x) {
            assert(x>=0);
            float const s = -0.5f - (x - floor(x));
            assert(s < 0 && s > -2.f);
            return s > -1.f ? s : s + 1.f;
        };
        for(float i = -radius + startOffset(location.y); i <= radius; i++){
            const float y = location.y + i;
            if (!in_range(y, 1.f, layerSize.y - 1.f)) {
                continue;
            }
            for (float j = -radius + (int) g.thread_rank() + startOffset(location.x); j <= radius; j += thrds_per_pt) {
                const float x = location.x + j;
                if (!in_range(x, 1.f, layerSize.x - 1.f)) {
                    continue;
                }
                // Note that we rotate with -angle here! OpenCV uses ori = 2*pi-angle
                const float c_rot = j * cos_t - i * sin_t;
                const float r_rot = j * sin_t + i * cos_t;
                if( in_range<float>(r_rot, -SIFT_DESCR_WIDTH/2, SIFT_DESCR_WIDTH/2)
                    && in_range<float>(c_rot, -SIFT_DESCR_WIDTH/2, SIFT_DESCR_WIDTH/2))
                {
                    const float rbin = r_rot + SIFT_DESCR_WIDTH/2 - 0.5f;
                    const float cbin = c_rot + SIFT_DESCR_WIDTH/2 - 0.5f;
                    const float dx = tex2D<float>(layer, x+1, y) - tex2D<float>(layer, x-1, y);
                    const float dy = tex2D<float>(layer, x, y+1) - tex2D<float>(layer, x, y-1);
                    const float theta = [&](){
                        // OpenCV uses atan2(-dy,dx) and minuses ori
                        float tmp = angle - atan2(dy, dx);
                        return tmp;
                    }();
                    const float magnitude = sqrtf(dx * dx + dy * dy);
                    const float weight = expf((c_rot * c_rot + r_rot * r_rot)*exp_scale);
                    const float delta = magnitude * weight;

                    const float obin = theta * (SIFT_DESCR_HIST_BINS / (2 * float(M_PI)));
                    int r0, c0, o0;
                    float rbin_res, cbin_res, obin_res;
                    {
                        const float tmp_r = floor(rbin);
                        const float tmp_c = floor(cbin);
                        const float tmp_o = floor(obin);
                        r0 = int(tmp_r);
                        c0 = int(tmp_c);
                        o0 = int(tmp_o);
                        rbin_res = rbin - tmp_r;
                        cbin_res = cbin - tmp_c;
                        obin_res = obin - tmp_o;
                    }

                    // histogram update using tri-linear interpolation
                    float v_r1 = delta*rbin_res, v_r0 = delta - v_r1;
                    float v_rc11 = v_r1*cbin_res, v_rc10 = v_r1 - v_rc11;
                    float v_rc01 = v_r0*cbin_res, v_rc00 = v_r0 - v_rc01;
                    float v_rco111 = v_rc11*obin_res, v_rco110 = v_rc11 - v_rco111;
                    float v_rco101 = v_rc10*obin_res, v_rco100 = v_rc10 - v_rco101;
                    float v_rco011 = v_rc01*obin_res, v_rco010 = v_rc01 - v_rco011;
                    float v_rco001 = v_rc00*obin_res, v_rco000 = v_rc00 - v_rco001;

#if 1
                    auto updateHist = [&](int r, int c, int o, float v){
                        if (in_range(r, 0, d) && in_range(c, 0, d)){
                            atomicAdd(&hist_grp[r][c][(o+n*2)%n], float2fixed(v));
                        }
                    };

                    const int r1 = r0 + 1, c1 = c0 + 1, o1 = o0 + 1;
                    updateHist(r0, c0, o0, v_rco000);
                    updateHist(r0, c0, o1, v_rco001);
                    updateHist(r0, c1, o0, v_rco010);
                    updateHist(r0, c1, o1, v_rco011);
                    updateHist(r1, c0, o0, v_rco100);
                    updateHist(r1, c0, o1, v_rco101);
                    updateHist(r1, c1, o0, v_rco110);
                    updateHist(r1, c1, o1, v_rco111);
                    // if (isDebug) {
                    //     printf("p=(%f,%f),i=%f,j=%f, dx=%f,dy=%f, angle=%f, rco0=(%d,%d,%d), rco1=(%d,%d,%d), v=(%f,%f,%f,%f,%f,%f,%f,%f)\n", x, y, i, j, dx, dy, angle, r0,c0,o0, r1,c1,o1, v_rco000, v_rco001, v_rco010, v_rco011, v_rco100, v_rco101, v_rco110, v_rco111);
                    // }
#else
                    // this seems to be wrong but gives better results
                    auto hist = [&hist_grp](int r, int c, int o)->uint32_t *{
                        return &hist_grp[(r+d)%d][(c+d)%d][(o+n)%n];
                    };
                    const int r1 = r0 + 1, c1 = c0 + 1, o1 = o0 + 1;
                    atomicAdd(hist(r0, c0, o0), float2fixed(v_rco000));
                    atomicAdd(hist(r0, c0, o1), float2fixed(v_rco001));
                    atomicAdd(hist(r0, c1, o0), float2fixed(v_rco010));
                    atomicAdd(hist(r0, c1, o1), float2fixed(v_rco011));
                    atomicAdd(hist(r1, c0, o0), float2fixed(v_rco100));
                    atomicAdd(hist(r1, c0, o1), float2fixed(v_rco101));
                    atomicAdd(hist(r1, c1, o0), float2fixed(v_rco110));
                    atomicAdd(hist(r1, c1, o1), float2fixed(v_rco111));
#endif
                }
            }
        }
        g.sync();
        float desc_thrd[elems_per_thrd];
#pragma unroll
        for (int i = 0; i < elems_per_thrd; i++) {
            const int k = i * g.size() + g.thread_rank();
            // const int k = g.thread_rank() * elems_per_thrd + i; // this make the desc like OpenCV
            if (k < SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS)
                desc_thrd[i] = fixed2float(hist_grp[0][g.thread_rank()][i]);
            else
                desc_thrd[i] = 0.f;
        }

        auto computeSum = [&](bool squared){
            float sum = 0.f;
            #pragma unroll
            for (int i = 0; i < elems_per_thrd; i++) {
                assert(desc_thrd[i] >= 0);
                sum += (squared ? sqr(desc_thrd[i]) : desc_thrd[i]);
            }
            sum = thread_group_sum<thrds_per_pt>(g, sum);
            return sum;
        };
        const float saturation_threshold = [&]() {
            auto const sqrSum = computeSum(true);
            return sqrt(sqrSum) * SIFT_DESCR_MAG_THR;
        }();
#pragma unroll
        for (int i = 0; i < elems_per_thrd; i++) {
            desc_thrd[i] = min(saturation_threshold, desc_thrd[i]);
        }
        const float desc_scale_factor = [&]() {
            float const sum = computeSum(!useRootSIFT);
            if (useRootSIFT) {
                return 1.f / std::max(sum, std::numeric_limits<float>::epsilon());
            }
            return SIFT_INT_DESCR_FCTR / std::max(sqrtf(sum), std::numeric_limits<float>::epsilon());
        }();
        static_assert(SIFT_DESCR_WIDTH * SIFT_DESCR_WIDTH * SIFT_DESCR_HIST_BINS % thrds_per_pt == 0, "fatal error");

        union{
            uint8_t desc[elems_per_thrd];
            float2 desc_cast;
        }desc_uchar;
        static_assert(sizeof(float2) == elems_per_thrd, "fatal error");
#pragma unroll
        for (int i = 0; i < elems_per_thrd; i++) {
            desc_thrd[i] *= desc_scale_factor;
            if (useRootSIFT) {
                desc_thrd[i] = sqrtf(desc_thrd[i]) * SIFT_INT_DESCR_FCTR;
            }
            assert(desc_thrd[i] >= 0);
            desc_uchar.desc[i] = uint8_t(std::min(255, int(desc_thrd[i])));
        }
        reinterpret_cast<float2*>(&descriptors[idx])[g.thread_rank()] = desc_uchar.desc_cast;
    }
}

//__global__ void kernel_describe_proxy(const cudaTextureObject_t* __restrict__ const gauss_layers,
//                          const int num_octave_layers,//layers per octave.
//                          const int num_octaves,
//                          const bool up_sampled,
//                          const GPUArray<KeyPoint>* __restrict__ const kpoints,
//                          const uint32_t max_num_kpoints,//min(max_num_kpoints, kpoints->count)
//                          const float max_radius,
//                          SiftDescriptor* const descriptors)
//{
//    const uint32_t num_kpoints = min(max_num_kpoints, kpoints->count);
//    if(num_kpoints != 0) {
//        const int grid_dim = div_ceil((int) num_kpoints, pts_per_cta);
//        kernel_describe << < grid_dim, block_dim >> >
//                                       (gauss_layers, num_octave_layers, num_octaves, up_sampled, kpoints->data, num_kpoints, max_radius, descriptors);
//    }
//}

cudaError_t cuda_describe(const cudaTextureObject_t* __restrict__ const gauss_layers,
                          const int num_octave_layers,//layers per octave.
                          const int num_octaves,
                          int2 const octave0_size, // up-scaled size if up-scaling is used.
                          const bool up_sampled,
                          const GPUArray<KeyPoint>* __restrict__ const kpoints,
                          const uint32_t num_kpoints,//min(max_num_kpoints, kpoints->count)
                          const float max_radius,
                          SiftDescriptor* const descriptors,
                          bool useRootSIFT,
                          const cudaStream_t stream)
{
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    if(num_kpoints != 0) {
        const int grid_dim = div_ceil((int) num_kpoints, pts_per_cta);
        if (useRootSIFT) {
            kernel_describe<true> << < grid_dim, block_dim, 0, stream >> >
                                       (gauss_layers, num_octave_layers, num_octaves, float2{(float)octave0_size.x, (float)octave0_size.y}, up_sampled, kpoints->data, num_kpoints, max_radius, descriptors);
        }
        else {
            kernel_describe<false> << < grid_dim, block_dim, 0, stream >> >
                                       (gauss_layers, num_octave_layers, num_octaves, float2{(float)octave0_size.x, (float)octave0_size.y}, up_sampled, kpoints->data, num_kpoints, max_radius, descriptors);
        }
    }
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    return cudaGetLastError();
}