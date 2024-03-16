//
// Created by yao on 13/01/18.
//

#ifdef __CLION_IDE__
#define __CUDACC__
#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <array>
#include "kernels.h"
#include "solve_GaussElim.h"
#include "utils_host.h"
#include <cooperative_groups.h>

#define DEBUG_INSPECT_POINT 0

namespace cg = cooperative_groups;

constexpr int2 kernel_find_extrema_cta_size = {32, 8};
constexpr int2 kernel_find_extrema_tile_size = {kernel_find_extrema_cta_size.x * 2, kernel_find_extrema_cta_size.y * 2};
__global__ void kernel_find_extrema(
        const cudaTextureObject_t* __restrict__ DoG_layers, const int num_DoG_layers,
        const int octave_id, const int2 img_size,
        GPUArray<ScaleSpaceExtrema>* const extremas, const uint32_t max_num_extremas,
        const float threshold){
    constexpr int2 block_dim = kernel_find_extrema_cta_size;
    constexpr int2 tile_size = kernel_find_extrema_tile_size;
    constexpr int2 tile_size_padded = {tile_size.x + 2, tile_size.y + 2};

    static_assert(tile_size_padded.x % 2 == 0, "fatal error");
    __shared__ float2 tile_layer_float2[tile_size_padded.y][tile_size_padded.x / 2];//use float2 to guarantee alignment
    float(&tile_layer)[tile_size_padded.y][tile_size_padded.x] = *reinterpret_cast<float(*)[tile_size_padded.y][tile_size_padded.x]>(&tile_layer_float2[0][0]);

    const int2 thrd_id = {(int)threadIdx.x, (int)threadIdx.y};
    const float2 tile_base = {float(tile_size.x * blockIdx.x) + 0.5f,
                              float(tile_size.y * blockIdx.y) + 0.5f};

    union RegTile{
        float data[4][4];
        float2 data_float2[4][2];
    };

    auto load_layer_shared = [&](const cudaTextureObject_t& layer, RegTile& dst) {
        __syncthreads();
        auto load_rows = [&](int i){
            const float y = tile_base.y + block_dim.y * i + thrd_id.y;
#pragma unroll
            for (int k = 0; k < tile_size_padded.x / block_dim.x; k++) {
                assert(block_dim.y * i + thrd_id.y < tile_size_padded.y && block_dim.x * k + thrd_id.x < tile_size_padded.x);
                tile_layer[block_dim.y * i + thrd_id.y][block_dim.x * k + thrd_id.x] = tex2D<float>(
                        layer,
                        tile_base.x + block_dim.x * k + thrd_id.x,
                        y);
            }
            const int k = tile_size_padded.x / block_dim.x;
            if (thrd_id.x < tile_size_padded.x % block_dim.x) {
                assert(block_dim.y * i + thrd_id.y < tile_size_padded.y && block_dim.x * k + thrd_id.x < tile_size_padded.x);
                tile_layer[block_dim.y * i + thrd_id.y][block_dim.x * k + thrd_id.x] = tex2D<float>(
                        layer,
                        tile_base.x + block_dim.x * k + thrd_id.x,
                        y);
            }
        };
        for (int i = 0; i < tile_size_padded.y / block_dim.y; i++) {
            load_rows(i);
        }
        if(thrd_id.y < tile_size_padded.y % block_dim.y){
            load_rows(tile_size_padded.y / block_dim.y);
        }
        __syncthreads();
        for(int i = thrd_id.y; i < tile_size_padded.y; i += block_dim.y){
            for(int j = thrd_id.x; j < tile_size_padded.x; j += block_dim.x){
                assert(tile_layer[i][j] == tex2D<float>(layer, tile_base.x + j, tile_base.y + i));
            }
        }

#pragma unroll
        for(int i = 0; i < 4; i++){
#pragma unroll
            for(int j = 0; j < 2; j++){
                assert(thrd_id.y * 2 + i < tile_size_padded.y && thrd_id.x + j < tile_size_padded.x / 2);
                dst.data_float2[i][j] = tile_layer_float2[thrd_id.y * 2 + i][thrd_id.x + j];

                assert(dst.data[i][j*2] == tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j * 2]);
                assert(dst.data[i][j*2 + 1] == tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j * 2 + 1]);
            }
//            for(int j = 0; j < 4; j++){
//                dst.data[i][j] = tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j];
//                assert(tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j] == tex2D<float>(layer, tile_base.x + thrd_id.x * 2 + j, tile_base.y + thrd_id.y * 2 + i));
//            }
            for(int j = 0; j < 4; j++) {
                assert(thrd_id.y * 2 + i < tile_size_padded.y && thrd_id.x * 2 + j < tile_size_padded.x);
                assert(dst.data[i][j] == tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j]);
            }
        }
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++) {
                assert(thrd_id.y * 2 + i < tile_size_padded.y && thrd_id.x * 2 + j < tile_size_padded.x);
                assert(dst.data[i][j] == tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j]);
                assert(tile_layer[thrd_id.y * 2 + i][thrd_id.x * 2 + j] == tex2D<float>(layer, tile_base.x + thrd_id.x * 2 + j, tile_base.y + thrd_id.y * 2 + i));
                assert(dst.data[i][j] == tex2D<float>(layer, tile_base.x + thrd_id.x * 2 + j, tile_base.y + thrd_id.y * 2 + i));
            }
        }
    };

    auto load_layer_direct = [&](const cudaTextureObject_t& layer, RegTile& dst){
        for(int i = 0; i < 4; i++){
            for(int j = 0; j < 4; j++){
                dst.data[i][j] = tex2D<float>(layer, tile_base.x + thrd_id.x * 2 + j, tile_base.y + thrd_id.y * 2 + i);
            }
        }
    };

	static_cast<void>(load_layer_shared);
//    auto load_layer = load_layer_shared;
    auto load_layer = load_layer_direct;//seems load_layer_direct is faster

    RegTile tile_prev_reg, tile_curr_reg;
    {//data load of the first two layers
        load_layer(DoG_layers[0], tile_prev_reg);
        load_layer(DoG_layers[1], tile_curr_reg);
    }

    __shared__ cudaTextureObject_t layer;
    for(int n = 1; n < num_DoG_layers - 1; n++){
        RegTile tile_next_reg;
        if(thrd_id.x == 0 && thrd_id.y == 0){
            layer = DoG_layers[n + 1];
        }
        __syncthreads();
        load_layer(layer, tile_next_reg);
//        load_layer(DoG_layers[n + 1], tile_next_reg);

        float(&prev)[4][4] = tile_prev_reg.data;
        float(&curr)[4][4] = tile_curr_reg.data;
        float(&next)[4][4] = tile_next_reg.data;
        {//detect phase
            auto gt = [](float a, float b){return a >= b;};
            auto lt = [](float a, float b){return a <= b;};
            auto check_extrema = [&](int i, int j, auto cmp){
                const auto centre = curr[i][j];
                if (std::abs(centre) < threshold)
                    return false;
                for(int pi = i - 1; pi < i + 2; pi++){

#if 0
                    for(int pj = j - 1; pj < j + 2; pj++){

                        if((pi != i || pj != j) && cmp(curr[pi][pj], centre))
                            return false;
                        if(cmp(prev[pi][pj], centre))
                            return false;
                        if(cmp(next[pi][pj], centre))
                            return false;
                    }
#else
                    bool not_extrema = false;
                    for(int pj = j - 1; pj < j + 2; pj++){
                        not_extrema |= (((pi != i || pj != j) && cmp(curr[pi][pj], centre)) | cmp(prev[pi][pj], centre) | cmp(next[pi][pj], centre));
                    }
                    if (not_extrema)
                        return false;
#endif
                }
                return true;
            };
#pragma unroll
            for(int i = 1; i < 3; i++){
#pragma unroll
                for(int j = 1; j < 3; j++){
                    const bool is_min = (curr[i][j] < 0 && check_extrema(i, j, lt));
                    // optimize the more frequent path:
                    // due to rareness of is_max and SIMT, it does not worth the effort to skip check_min when is_max is true.
                    const bool is_max = (curr[i][j] > 0 && check_extrema(i, j, gt));
                    if(is_max || is_min){
                        assert(is_max != is_min);
                        ScaleSpaceExtrema extrema;
                        extrema.location = {
                                ushort(tile_size.x * blockIdx.x + threadIdx.x * 2 + j),
                                ushort(tile_size.y * blockIdx.y + threadIdx.y * 2 + i)
                        };
                        extrema.octave = uint8_t(octave_id);
                        extrema.layer = uint8_t(n);
                        extrema.is_max = is_max;
                        extremas->push_back(extrema, max_num_extremas);
                        assert(curr[i][j] == tex2D<float>(DoG_layers[n], extrema.location.x + 0.5f, extrema.location.y + 0.5f));
                    }
                }
            }
        }
#pragma unroll
        for(int i = 0; i < 4; i++){
#pragma unroll
            for(int j = 0; j < 4; j++){
                prev[i][j] = curr[i][j];
                curr[i][j] = next[i][j];
            }
        }
    }
}

cudaError_t cuda_find_extrema(
        const cudaTextureObject_t *const DoG_layers, const int num_DoG_layers,
        const int octave_id, const int2 img_size,
        GPUArray<ScaleSpaceExtrema> *const extremas, const uint32_t max_num_extremas, const float threshold,
        const bool reset_extremas,
        const cudaStream_t stream){
    if(reset_extremas) {
#if 0
        kernel_memset_scalar<<<1,1,0, stream>>>(&candidates->count);
        const cudaError_t err = cudaGetLastError();
#else
        const cudaError_t err = cudaMemsetAsync(&extremas->count, 0, sizeof(&extremas->count), stream);
#endif
        if(err != cudaSuccess)
            return err;
    }
    const dim3 dimBlock((unsigned)kernel_find_extrema_cta_size.x, (unsigned)kernel_find_extrema_cta_size.y);
    const dim3 dimGrid((unsigned)div_ceil(img_size.x, kernel_find_extrema_tile_size.x), (unsigned)div_ceil(img_size.y, kernel_find_extrema_tile_size.y));
    kernel_find_extrema<<<dimGrid, dimBlock, 0, stream>>>(DoG_layers, num_DoG_layers, octave_id, img_size, extremas, max_num_extremas, threshold);

//    checkCudaError(cudaDeviceSynchronize());
    return cudaGetLastError();
}

__global__ void kernel_make_keypoints(
        const GPUArray<ScaleSpaceExtrema>* __restrict__ const extremas, const uint32_t num_extremas,
        const cudaTextureObject_t* __restrict__ DoG_layers, const int num_DoG_layers,
        const int octave_id,//for assertions only
        const int2 img_size,
        const int2 loc_lb, const int2 loc_ub,
        const float2 loc_lb_fp, const float2 loc_ub_fp,
        const float thres_contrast, const float thres_edge,
        const float sigma,
        GPUArray<KeyPoint>* kpoints, const uint32_t max_num_kpoints)
{
    (void)num_DoG_layers; (void)octave_id;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

//    if(idx < extremas->count)
    if(idx < num_extremas)
//    for(int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < extremas->count; idx += blockDim.x * gridDim.x)
    {
        const ScaleSpaceExtrema extrema = extremas->data[idx];
        assert(extrema.octave == octave_id);
        assert(extrema.layer >= 1 && extrema.layer < num_DoG_layers - 1);

        int3 loc_extrema = {extrema.location.x, extrema.location.y, extrema.layer};
        for (int iteration = 0; iteration < SIFT_MAX_INTERP_STEPS; iteration++) {
            float neighbour[3][3][3];
            const float2 xy_base = {loc_extrema.x - 0.5f, loc_extrema.y - 0.5f};// +0.5 - 1 because x and y start from 0, not -1
            for (int z = 0; z < 3; z++) {
                const cudaTextureObject_t layer = DoG_layers[loc_extrema.z - 1 + z];
                for (int y = 0; y < 3; y++) {
                    for (int x = 0; x < 3; x++) {
                        neighbour[z][y][x] = tex2D<float>(layer, xy_base.x + x, xy_base.y + y);
                    }
                }
            }
            //assert it is an extrema
            if (iteration == 0) {
                for (int z = 0; z < 3; z++) {
                    for (int y = 0; y < 3; y++) {
                        for (int x = 0; x < 3; x++) {
                            if (x != 1 || y != 1) {
                                assert(extrema.is_max ? neighbour[z][y][x] <= neighbour[1][1][1] :
                                       neighbour[z][y][x] >=
                                       neighbour[1][1][1]);
                            }
                        }
                    }
                }
            }
            const float b[3] = {
                    (neighbour[1][1][2] - neighbour[1][1][0]) * 0.5f,
                    (neighbour[1][2][1] - neighbour[1][0][1]) * 0.5f,
                    (neighbour[2][1][1] - neighbour[0][1][1]) * 0.5f
            };
            const float dxx = neighbour[1][1][0] + neighbour[1][1][2] - neighbour[1][1][1] * 2;
            const float dyy = neighbour[1][0][1] + neighbour[1][2][1] - neighbour[1][1][1] * 2;
            const float dzz = neighbour[0][1][1] + neighbour[2][1][1] - neighbour[1][1][1] * 2;
            const float dxy =
                    ((neighbour[1][2][2] - neighbour[1][2][0]) - (neighbour[1][0][2] - neighbour[1][0][0])) * 0.25f;
            const float dyz =
                    ((neighbour[2][2][1] - neighbour[2][0][1]) - (neighbour[0][2][1] - neighbour[0][0][1])) * 0.25f;
            const float dzx =
                    ((neighbour[2][1][2] - neighbour[0][1][2]) - (neighbour[2][1][0] - neighbour[0][1][0])) * 0.25f;
            const float A[3][3] = {
                    dxx, dxy, dzx,
                    dxy, dyy, dyz,
                    dzx, dyz, dzz,
            };
            float X[3];
            solve_GaussElim<float>(A, b, X);//@todo: check correctness

            // Originally just int(round(x)). Use lazyDelta to avoid hopping back and forth
            auto lazyDelta = [](float x){
                // return int(round(x));
                return std::abs(x) < 1.f ? 0 : int(round(x));
            };
            const int3 delta = {
                    lazyDelta(-X[0]),
                    lazyDelta(-X[1]),
                    lazyDelta(-X[2])
            };
#if DEBUG_INSPECT_POINT
            bool isDebug = false;
            if (1) {//iteration converged. make kpoint and break.
                const float3 loc_delta = {-X[0], -X[1], -X[2]};
                const float3 loc_extrema_prec = {
                        loc_extrema.x + loc_delta.x,
                        loc_extrema.y + loc_delta.y,
                        loc_extrema.z + loc_delta.z
                };

                KeyPoint kpoint;
                kpoint.octave = int8_t(extrema.octave);//or uint8_t(octave_id)
                kpoint.layer = uint8_t(round(loc_extrema_prec.z));
                kpoint.layer_offset = (int8_t) round((loc_extrema_prec.z - kpoint.layer) * 256 - (0.5f));
                // we leave location in current octave resolution. will convert to original resolution during orientation assignment
                kpoint.location = {
                        loc_extrema_prec.x + 0.5f,// * (1 << kpoint.octave),
                        loc_extrema_prec.y + 0.5f// * (1 << kpoint.octave)
                };

                //reject low-contract points
                //@fixme: why the 0.5f? opencv is doing this
                const float contrast =
                        neighbour[1][1][1] + (loc_delta.x * b[0] + loc_delta.y * b[1] + loc_delta.z * b[2]) * 0.5f;
                kpoint.response = __float2half(std::abs(contrast));

                const float kpoint_size =
                        sigma * powf(2.f, loc_extrema_prec.z / float(num_DoG_layers - 2))/* * (1 << kpoint.octave)*/ * 2;
                kpoint.size = __float2half(kpoint_size);
                
                bool const octave0_is_up_sampled = false;
                auto& kpt = kpoint;
                const float scale = float(1 << kpt.octave) * (octave0_is_up_sampled ? 0.5f : 1.f);
                if (octave0_is_up_sampled)
                    kpt.octave -= 1;//now octave index start from -1, rather than 0
                kpt.location.x *= scale;
                kpt.location.y *= scale;
                kpt.size = __float2half(__half2float(kpt.size) * scale);
                float2 const refPt = {982.306458, 440.357819};
                if (sqr(kpt.location.x - refPt.x) + sqr(kpt.location.y - refPt.y) < sqr(5.f)) {
                    isDebug = true;
                    printf("[rsift]: (%f,%f), size=%f, response=%f\n", kpt.location.x, kpt.location.y, __half2float(kpt.size), __half2float(kpt.response));
                }
            }
#endif //DEBUG_INSPECT_POINT
            if (/*iteration+1 == SIFT_MAX_INTERP_STEPS ||*/ delta.x == 0 && delta.y == 0 && delta.z == 0) {//iteration converged. make kpoint and break.
                const float3 loc_delta = {-X[0], -X[1], -X[2]};
                const float3 loc_extrema_prec = {
                        loc_extrema.x + loc_delta.x,
                        loc_extrema.y + loc_delta.y,
                        loc_extrema.z + loc_delta.z
                };
                //reject out-of-range points
                if (!in_range(loc_extrema_prec.x, loc_lb_fp.x, loc_ub_fp.x)
                    || !in_range(loc_extrema_prec.y, loc_lb_fp.y, loc_ub_fp.y)
                    || !in_range(loc_extrema_prec.z, 0.f, float(num_DoG_layers - 1)))
                {
#if DEBUG_INSPECT_POINT
                    if (isDebug) {
                        printf("reject out-of-range\n");
                    }
#endif // DEBUG_INSPECT_POINT
                    break;
                }

                KeyPoint kpoint;
                kpoint.octave = int8_t(extrema.octave);//or uint8_t(octave_id)
                kpoint.layer = uint8_t(round(loc_extrema_prec.z));
                kpoint.layer_offset = (int8_t) round((loc_extrema_prec.z - kpoint.layer) * 256 - (0.5f));
                // we leave location in current octave resolution. will convert to original resolution during orientation assignment
                kpoint.location = {
                        loc_extrema_prec.x + 0.5f,// * (1 << kpoint.octave),
                        loc_extrema_prec.y + 0.5f// * (1 << kpoint.octave)
                };

                //reject low-contract points
                //@fixme: why the 0.5f? opencv is doing this
                const float contrast =
                        neighbour[1][1][1] + (loc_delta.x * b[0] + loc_delta.y * b[1] + loc_delta.z * b[2]) * 0.5f;
                if (std::abs(contrast) * (num_DoG_layers - 2) < thres_contrast) {
#if DEBUG_INSPECT_POINT
                    if (isDebug) {
                        printf("reject low contrast\n");
                    }
#endif // DEBUG_INSPECT_POINT
                    return;
                }
                kpoint.response = __float2half(std::abs(contrast));

                //reject edge points
                const float tr = dxx + dyy;
                const float det = dxx * dyy - dxy * dxy;
                if (det <= 0 || sqr(tr) * thres_edge >= sqr(thres_edge + 1) * det) {
#if DEBUG_INSPECT_POINT
                    if (isDebug) {
                        printf("reject edge point\n");
                    }
#endif // DEBUG_INSPECT_POINT
                    return;
                }
                const float kpoint_size =
                        sigma * powf(2.f, loc_extrema_prec.z / float(num_DoG_layers - 2))/* * (1 << kpoint.octave)*/ * 2;
                kpoint.size = __float2half(kpoint_size);

                kpoints->push_back(kpoint, max_num_kpoints);
                break;
            } else {//not converged, prepare for next iteration
                loc_extrema = {
                    loc_extrema.x + delta.x,
                    loc_extrema.y + delta.y,
                    loc_extrema.z + delta.z
                };
                if (!in_range(loc_extrema.x, loc_lb.x, loc_ub.x)
                    || !in_range(loc_extrema.y, loc_lb.y, loc_ub.y)
                    || !in_range(loc_extrema.z, 1, num_DoG_layers-1))
                {
#if DEBUG_INSPECT_POINT
                    if (isDebug) {
                        printf("reject out-of-bound\n");
                    }
#endif // DEBUG_INSPECT_POINT
                    return;
                }
            }
        }
    }
}

cudaError_t cuda_get_num_extremas(
        const GPUArray<ScaleSpaceExtrema>* const extremas, uint32_t* num_extremas, const cudaStream_t stream){
    return cudaMemcpyAsync(num_extremas, &extremas->count, 4, cudaMemcpyDeviceToHost, stream);
}

cudaError_t cuda_make_keypoints(const GPUArray<ScaleSpaceExtrema>* const extremas, const uint32_t num_extremas,
                                const cudaTextureObject_t* __restrict__ DoG_layers, const int num_DoG_layers,
                                const int octave_id,//for assertions only
                                const int2 img_size,
                                const int2 loc_lb, const int2 loc_ub,
                                const float thres_contrast, const float thres_edge,
                                const float sigma,
                                GPUArray<KeyPoint>* const kpoints, const uint32_t max_num_kpoints,
                                const bool reset_kpoints, const cudaStream_t stream)
{
    cudaError_t err = cudaSuccess;

    if(reset_kpoints) {
#if 0
        kernel_memset_scalar<<<1,1,0, stream>>>(&candidates->count);
        err = cudaGetLastError();
#else
        err = cudaMemsetAsync(&kpoints->count, 0, sizeof(&kpoints->count), stream);
#endif
    }
    if(err != cudaSuccess)
        return err;
    if(num_extremas > 0) {
#ifdef VERBOSE
        printf("octave #%d: %u/%u extremas\n", octave_id, num_extremas);
#endif
        const int grid_dim = div_ceil(num_extremas, 128u);
        kernel_make_keypoints << < grid_dim, 128, 0, stream>> >
                                             (extremas, num_extremas, DoG_layers, num_DoG_layers, octave_id, img_size,
                                                     loc_lb, loc_ub, float2{(float) loc_lb.x, (float) loc_lb.y}, float2{
                                                     (float) loc_ub.x - 1, (float) loc_ub.y - 1},
                                                     thres_contrast, thres_edge, sigma, kpoints, max_num_kpoints);
    }else{
#ifdef VERBOSE
        printf("no extremas in octave %d\n", octave_id);
#endif
    }
    return cudaGetLastError();
}

cudaError_t cuda_get_num_kpoints(
        const GPUArray<KeyPoint>* kpoints, uint32_t* num_kpoints, cudaStream_t stream){
    return cudaMemcpyAsync(num_kpoints, &kpoints->count, 4, cudaMemcpyDeviceToHost, stream);
}