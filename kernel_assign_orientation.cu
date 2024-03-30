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
// Created by yao on 15/01/18.
//

#ifdef __CLION_IDE__
#define __CUDACC__
#endif

#include "kernels.h"
#include "conv1d.h"
#include "utils_host.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

static constexpr int pts_per_cta = 16;
static constexpr int thrds_per_pt = 8;
static constexpr int block_dim = pts_per_cta * thrds_per_pt;
static __constant__ float bin_smooth_filter[5] = {1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16};
//static __constant__ float bin_smooth_filter[5] = {0, 0, 1, 0, 0};
// also corrects kpoint location to original resolution image
__global__ void kernel_assign_orientation(
        const cudaTextureObject_t* __restrict__ const gauss_layers,
        const int num_octave_layers,//layers per octave.
        const int num_octaves,
        float2 const octave0_size, // up-scaled size if up-scaling is used.
        GPUArray<KeyPoint>* const kpoints, const uint32_t max_num_kpoints, //@info: splited new key points are always appended at the end.
        const uint32_t num_points_before_split, bool octave0_is_up_sampled)
{
    const cg::thread_block_tile<thrds_per_pt> g = cg::tiled_partition<thrds_per_pt>(cg::this_thread_block());
    const int idx = pts_per_cta * blockIdx.x + threadIdx.x / g.size();// (block_dim * cg::this_thread_block().group_index().x + cg::this_thread_block().thread_index().x) / g.size();
    if(idx < min(num_points_before_split, max_num_kpoints))
//    for(int idx = cg::this_grid().thread_rank() / g.size(); idx < min(*num_points_before_split, max_num_kpoints); idx += cg::this_grid().size() / g.size())
    {
        KeyPoint kpt = kpoints->data[idx];
#if 1
        const float layerDelta = (kpt.layer_offset + 0.5f) * (1.f / 256);
        int32_t const idxGaussLayerInOctave = (int)floor(kpt.layer + layerDelta);
        assert(in_range(idxGaussLayerInOctave, 0, num_octave_layers + 1));
        const cudaTextureObject_t layer = gauss_layers[(num_octave_layers + 1) * kpt.octave + idxGaussLayerInOctave];
#else
        const cudaTextureObject_t layer = gauss_layers[num_octave_layers * kpt.octave + (kpt.layer - 1)];
#endif
        const float scale_octave = __half2float(kpt.size) * 0.5f;// / (1 << kpt.octave);
        const int radius = (int) round(SIFT_ORI_RADIUS * scale_octave);
        const float sigma = SIFT_ORI_SIG_FCTR * scale_octave;

        constexpr int num_bins = SIFT_ORI_HIST_BINS;
//    __shared__ float bins_cta[pts_per_cta][round_up(num_bins - thrds_per_pt, 32) + thrds_per_pt];
        const int smooth_halo = 2;
        __shared__ float bins_cta[pts_per_cta][num_bins + smooth_halo * 2 + 1];//1 is for bank conflict resolving.
        float(&bins)[num_bins] = *reinterpret_cast<float (*)[num_bins]>(&bins_cta[threadIdx.x / g.size()][smooth_halo]);
        if (g.thread_rank() == 0) {
            for (int i = 0; i < num_bins; i++)
                bins[i] = 0.f;
        }
        g.sync();

        float2 const layerSize = {octave0_size.x / (1<<kpt.octave), octave0_size.y / (1<<kpt.octave)};
        const float expf_scale = -1.f / (2.f * sigma * sigma);
        auto startOffset = [] (float x) {
            assert(x>=0);
            float const s = -0.5f - (x - floor(x));
            assert(s < 0 && s > -2.f);
            return s > -1.f ? s : s + 1.f;
        };
        for (float i = -radius + startOffset(kpt.location.y); i < radius + 0.99f; i++) {
            const float y = kpt.location.y + i;
            if (!in_range(y, 1.f, layerSize.y - 1.f)) {
                continue;
            }
            for (float j = -radius + startOffset(kpt.location.x) + (int) g.thread_rank(); j < radius + 0.99f; j += g.size()) {
                const float x = kpt.location.x + j;
                if (!in_range(x, 1.f, layerSize.x - 1.f)) {
                    continue;
                }
                // OpenCV does not do this. But seems it does no harm and no improvement
                if (sqr(i) + sqr(j) > sqr(radius)) {
                    continue;
                }
                const float dx = tex2D<float>(layer, x + 1, y) - tex2D<float>(layer, x - 1, y);
                const float dy = tex2D<float>(layer, x, y + 1) - tex2D<float>(layer, x, y - 1);
//                printf("dx = %f, dy = %f\n", dx, dy);
                const float theta = atan2(dy, dx);
                const int idx_bin = [&]() {
                    int tmp = (int) round(theta * float(num_bins / (2 * M_PI)));
                    return tmp < 0 ? tmp + num_bins : tmp;
                }();
                const float magnitude = sqrtf(dx * dx + dy * dy);
                const float weight = expf(expf_scale * (i * i + j * j));
                assert(idx_bin >= 0 && idx_bin < num_bins);
//                printf("adding %f to bins[%d] of pt = %d of cta %u\n", magnitude * weight, idx_bin, threadIdx.x / g.size(), blockIdx.x);
                atomicAdd(&bins[idx_bin],
                          magnitude * weight);//@fixme: might be slow because no hardware smem float atomicAdd
            }
        }
        g.sync();
        //smooth the bins
        constexpr int smooth_size = smooth_halo * 2 + 1;
        if (g.thread_rank() == 0) {
            float max_bin = 0.f;
            int idx_max_bin = 0;
            for (int i = 0; i < smooth_halo; i++) {
                bins[-smooth_halo + i] = bins[num_bins - smooth_halo + i];
                bins[num_bins + i] = bins[i];
            }
            conv1d<2> conv;
            conv.set_filter(bin_smooth_filter);
            float(&bin_grps)[div_ceil(num_bins + 4,smooth_size)][smooth_size] = *reinterpret_cast<
                    float (*)[div_ceil(num_bins + 4,smooth_size)][smooth_size]>(
                    &bins_cta[threadIdx.x / g.size()][0]);//or &bins[-smooth_halo]
            conv.init(bin_grps[0]);
            for (int i = 1; i < (num_bins + 4) / smooth_size; i++) {
                float conv_out[smooth_size];
                conv(bin_grps[i], conv_out, smooth_size);
                for (int j = 0; j < smooth_size; j++) {
                    bin_grps[i - 1][j] = conv_out[j];
                    if (conv_out[j] > max_bin) {
                        max_bin = conv_out[j];
                        idx_max_bin = i * smooth_size - smooth_size + j;
                    }
                }
            }
            {
                const int i = (num_bins + 4) / smooth_size;
                float conv_out[smooth_size];
                const int remain = num_bins % smooth_size;
                conv(bin_grps[i], conv_out, remain);
                for (int j = 0; j < smooth_size; j++) {
                    if (j < remain) {
                        bin_grps[i - 1][j] = conv_out[j];
                        if (conv_out[j] > max_bin) {
                            max_bin = conv_out[j];
                            idx_max_bin = i * smooth_size - smooth_size + j;
                        }
                    }
                }
            }

            {
                const float scale = float(1 << kpt.octave) * (octave0_is_up_sampled ? 0.5f : 1.f);
                if (octave0_is_up_sampled)
                    kpt.octave -= 1;//now octave index start from -1, rather than 0
                kpt.location.x *= scale;
                kpt.location.y *= scale;
                kpt.size = __float2half(__half2float(kpt.size) * scale);
            }
            assert(max_bin > 0.f);
            const float thres_split = max_bin * SIFT_ORI_PEAK_RATIO;
            float(&bins_smoothed)[num_bins] = *reinterpret_cast<float(*)[num_bins]>(&bin_grps[0][0]);
            for (int i = 0; i < num_bins; i++) {
                const int left = (i == 0 ? num_bins - 1 : i - 1);
                const int right = (i == num_bins - 1 ? 0 : i + 1);
                if (bins_smoothed[i] >= bins_smoothed[left] // must be >=, otherwise there are rare cases where some kpoints are not updated.
                    && bins_smoothed[i] >= bins_smoothed[right]) {
                    const float bin = i + 0.5f * (bins_smoothed[right] - bins_smoothed[left]) / (bins_smoothed[i] * 2 - bins_smoothed[left] - bins_smoothed[right]);
                    kpt.angle = uint8_t(((int)round(bin * (256.f / num_bins)) + 256) % 256);
                    if(i == idx_max_bin) {
                        kpoints->data[idx] = kpt;
                    }
                    else if (bins_smoothed[i] > thres_split)
                        kpoints->push_back(kpt, max_num_kpoints);
                }
            }
        }
    }
}

//__global__ void kernel_assign_orientation_proxy(
//        const cudaTextureObject_t* __restrict__ const gauss_layers,
//        const int num_octave_layers,//layers per octave.
//        const int num_octaves,
//        GPUArray<KeyPoint>* const kpoints, const uint32_t max_num_kpoints, bool octave0_is_up_sampled)
//{
//    if(kpoints->count > 0) {
//#ifdef VERBOSE
//        printf("num_kpoints = %u\n", kpoints->count);
//#endif
//        const int grid_dim = div_ceil((int) kpoints->count, pts_per_cta);
//        kernel_assign_orientation << < grid_dim, block_dim >> >
//                                                 (gauss_layers, num_octave_layers, num_octaves,
//                                                         kpoints, max_num_kpoints, kpoints->count, octave0_is_up_sampled);
//    }else{
//#ifdef VERBOSE
//        printf("no key points found\n");
//#endif
//    }
//}

// also corrects kpoint location to original resolution image
cudaError_t cuda_assign_orientation(
        const cudaTextureObject_t* const gauss_layers,
        const int num_octave_layers,//layers per octave.
        const int num_octaves,
        int2 const octave0_size, // up-scaled size if up-scaling is used.
        GPUArray<KeyPoint>* const kpoints, const uint32_t max_num_kpoints, const uint32_t num_points_before_split,
        bool octave0_is_up_sampled,
        const cudaStream_t stream)
{
    if(num_points_before_split > 0) {
#ifdef VERBOSE
        printf("num_kpoints = %u\n", kpoints->count);
#endif
        const int grid_dim = div_ceil((int) num_points_before_split, pts_per_cta);
        kernel_assign_orientation << < grid_dim, block_dim, 0, stream >> >
                                                 (gauss_layers, num_octave_layers, num_octaves, float2{(float)octave0_size.x, (float)octave0_size.y},
                                                         kpoints, max_num_kpoints, num_points_before_split, octave0_is_up_sampled);
    }else{
#ifdef VERBOSE
        printf("no key points found\n");
#endif
    }

    return cudaGetLastError();
}