//
// Created by yao on 8/01/18.
//

#ifdef __CLION_IDE__
#define __CUDACC__
#endif

#include <texture_types.h>
#include <array>
#include "conv1d.h"
#include "types.h"
#include "utils_sift.h"
#include "utils_host.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include <vector>
#include <algorithm>

constexpr int CTA_size = 32;

template <int filter_size>
__device__ __forceinline__ void load_pixels(
        const int tile_x, const cudaTextureObject_t& tex, const float2& img_size_inv, const int2 base,
        float (&pixels)[filter_size], const int num_pixels = filter_size){
    static_assert(filter_size % 2 == 1, "fatal error");
    const float x = (float(base.x + tile_x) + 0.5f) * img_size_inv.x;
    const float y0 = (float(base.y) + 0.5f) * img_size_inv.y;
//#pragma unroll
    for(int i = 0; i < filter_size; i++){
        if(i >= num_pixels)
            break;
        pixels[i] = tex2D<float>(tex, x, y0 + i * img_size_inv.y);
    }
    __syncthreads();
}

template <typename T>
__device__ __forceinline__ void swap(T& a, T& b){
    T c = a;
    a = b;
    b = c;
}

template <int filter_radius, bool blur_only, bool DoG_only, int preload_next_tile, bool initial>
__device__ __forceinline__ void conv_tile(
        const int& cta_id, const int thrd_id,
        const cudaTextureObject_t& tex, const int2& img_size, const float2& img_size_inv,
        const int iter_tile,
        conv1d<filter_radius>& conv,
        float (&pixels)[CTA_size + filter_radius * 2],
        float (&buffer)[CTA_size][(filter_radius + CTA_size + preload_next_tile) | 1],
        float (&pixels_last)[CTA_size][preload_next_tile],
        const pitched_ptr<float>& DoG, const pitched_ptr<float>& blurred)
{
    constexpr int filter_size = filter_radius * 2 + 1;
    constexpr int tile_size_padded = CTA_size + filter_radius * 2;

    assert((initial && iter_tile < 0) || (!initial && iter_tile >= 0));
    const int2 tile_base = {iter_tile * CTA_size, cta_id * CTA_size};

    const int tile_x = (thrd_id < preload_next_tile ? thrd_id + CTA_size : thrd_id);

#define LOAD_ON_DEMAND 1
    //load pixels
#if !LOAD_ON_DEMAND
    {
        const float x = (float(tile_base.x + tile_x) + 0.5f) * img_size_inv.x;
        const float y0 = (float(tile_base.y) - filter_radius + 0.5f) * img_size_inv.y;
        for(int i = 0; i < tile_size_padded; i++){
            pixels[i] = tex2D<float>(tex, x, y0 + i * img_size_inv.y);
        }
    }
#endif
    //vertical conv
    {
        constexpr int nIters = tile_size_padded / filter_size + 1;
        float (&pixel_groups)[nIters][filter_size] = *reinterpret_cast<float (*)[nIters][filter_size]>(&pixels);
        static_assert(tile_size_padded > filter_size, "fatal error");
        {
            const int iter = 0;
            float(&pixel_grp)[filter_size] = pixel_groups[iter];
#if LOAD_ON_DEMAND
            const int2 base = {tile_base.x, tile_base.y - filter_radius + filter_size * iter};
            load_pixels(tile_x, tex, img_size_inv, base, pixel_grp, filter_size);
#endif
            conv.init(pixel_grp);
        }
//#pragma unroll
        for (int iter = 1; iter < tile_size_padded / filter_size; iter++) {
            float(&pixel_grp)[filter_size] = pixel_groups[iter];
#if LOAD_ON_DEMAND
            const int2 base = {tile_base.x, tile_base.y - filter_radius + filter_size * iter};
            load_pixels(tile_x, tex, img_size_inv, base, pixel_grp, filter_size);
#endif
            float conv_out[filter_size];
            //conv(pixel_grp, conv_out, filter_size);
            conv.compute<true>(pixel_grp, conv_out, filter_size);
            for (int i = 0; i < filter_size; i++) {
                buffer[filter_size * (iter - 1) + i][filter_radius + tile_x] = conv_out[i];
            }
        }
        {
            const int iter = tile_size_padded / filter_size;
            float(&pixel_grp)[filter_size] = pixel_groups[iter];
#if LOAD_ON_DEMAND
            const int2 base = {tile_base.x, tile_base.y - filter_radius + filter_size * iter};
            load_pixels(tile_x, tex, img_size_inv, base, pixel_grp, tile_size_padded % filter_size);
#endif
            float conv_out[filter_size];
            //conv(pixel_grp, conv_out, tile_size_padded % filter_size);
            conv.compute<false>(pixel_grp, conv_out, tile_size_padded % filter_size);
            for (int i = 0; i < filter_size; i++) {
                if (i > tile_size_padded % filter_size)//not >=, this is *not* an error
                    break;
                buffer[filter_size * (iter - 1) + i][filter_radius + tile_x] = conv_out[i];
            }
        }
    }
    __syncthreads();
    /*if(initial){//this is optional as the conv results of the initial tile is dropped anyway
        const float val_left = buffer[thrd_id][filter_radius * 2];
        for(int i = 0; i < filter_radius * 2; i++)
            buffer[thrd_id][i] = val_left;
    }*/

    //horizontal conv
    if(!initial)
    {
        constexpr int nIters = div_ceil(tile_size_padded, filter_size);
        float (&pixel_groups)[nIters][filter_size] = *reinterpret_cast<float (*)[nIters][filter_size]>(&buffer[thrd_id]);
        {
            const int iter = 0;
            conv.init(pixel_groups[iter]);
        }
        for (int iter = 1; iter < tile_size_padded / filter_size; iter++) {
#if 1
            conv(pixel_groups[iter], pixel_groups[iter - 1], filter_size);
#else
            float conv_out[filter_size];
            conv(pixel_groups[iter], conv_out, filter_size);
            for (int i = 0; i < filter_size; i++) {
                buffer[thrd_id][filter_size * (iter - 1) + i] = conv_out[i];
            }
#endif
        }
        {
            const int iter = tile_size_padded / filter_size;
#if 1
            conv(pixel_groups[iter], pixel_groups[iter - 1], tile_size_padded % filter_size);
#else
            float conv_out[filter_size];
            conv(pixel_groups[iter], conv_out, tile_size_padded % filter_size);
            for (int i = 0; i < filter_size; i++) {
                if (i > tile_size_padded % filter_size)//not >=, this is *not* an error
                    break;
                buffer[thrd_id][filter_size * (iter - 1) + i] = conv_out[i];
            }
#endif
        }
    }

    __syncthreads();

    float(&core_pixels)[CTA_size] = *reinterpret_cast<float(*)[CTA_size]>(&pixels[filter_radius]);

    //store right out-of-tile pixels and get left in-tile pixels
    if(thrd_id < preload_next_tile) {
//#pragma unroll
        for (int i = 0; i < CTA_size; i++) {
            swap(core_pixels[i], pixels_last[i][thrd_id]);
        }
    }
    //store DoG and blurred result into global memory
    if(!initial){
        const int x = iter_tile * CTA_size + thrd_id;
        if(in_range(x, 0, img_size.x)) {
            const int y0 = cta_id * CTA_size;
            uint8_t* p_blurred = DoG_only ? nullptr : reinterpret_cast<uint8_t*>(blurred.get_ptr(y0, x));
            uint8_t* p_DoG = reinterpret_cast<uint8_t*>(DoG.get_ptr(y0, x));
            const int ub_i = img_size.y - y0;
            assert(ub_i >= 0);
            auto store = [&](bool no_bound_check){
                //#pragma unroll
                for (int i = 0; i < CTA_size; i++) {
                    if(no_bound_check || i < ub_i) {
                        const float pixel_blurred = buffer[i][thrd_id];
                        if(!blur_only) {
                            *reinterpret_cast<float *>(p_DoG) = core_pixels[i] - pixel_blurred;
                            p_DoG += DoG.pitch;
                        }
                        if (!DoG_only){
                            *reinterpret_cast<float *>(p_blurred) = pixel_blurred;
                            p_blurred += blurred.pitch;
                        }
                    }
                }
            };
            if(ub_i < CTA_size)
                store(false);
            else
                store(true);
        }
    }
    __syncthreads();//the next operation will overwrite left side of buffer. so need sync. set CTA_size=64 to test
    //prepare data that can be reused by the next tile
//#pragma unroll
    for(int i = 0; i < filter_radius + preload_next_tile; i++){
        buffer[thrd_id][i] = buffer[thrd_id][CTA_size + i];
    }
    __syncthreads();//make sure reusable data is copied to the correct location before being overwritten by the next tile
}

template <int filter_radius, bool blur_only, bool DoG_only, bool keep_pixels_in_register>//down sample blurred image or not
__global__ void kernel_DoG(const cudaTextureObject_t tex, const int2 img_size, const float2 img_size_inv,
                           const std::array<float, filter_radius * 2 + 1> filter1d,//left half and centre. symmetric filter assumed
                           const pitched_ptr<float> DoG, const pitched_ptr<float> blurred)
{
    constexpr int preload_next_tile = filter_radius;//should be in range [filter_radius, CTA_size]
    static_assert(preload_next_tile >= filter_radius, "fatal error");

    assert(threadIdx.y == 0 && threadIdx.z == 0);
    assert(blockDim.x == CTA_size);
    const int thrd_id = threadIdx.x;

    __shared__ float buffer[CTA_size][(CTA_size + filter_radius + preload_next_tile) | 1];
    __shared__ float pixels_last[CTA_size][preload_next_tile];

    conv1d<filter_radius> conv;
    conv.set_filter(*reinterpret_cast<const float(*)[filter_radius * 2 + 1]>(&filter1d[0]));

    assert(gridDim.y == 1 && gridDim.z == 1);
    assert(gridDim.x == div_ceil(img_size.y, CTA_size));

    //use shared memory (need volatile?) cta_id to force conv_tile to recalculate y when loading pixels. This saves a lot of registers
    __shared__ int cta_id;
    if(threadIdx.x == 0)
        cta_id = blockIdx.x;
    __syncthreads();
//    const int cta_id = blockIdx.x;
    if(cta_id >= div_ceil(img_size.y, CTA_size))
        return;

    const int tiles_per_row = div_ceil(img_size.x, CTA_size);
//#if 0
    float pixels_register[CTA_size + filter_radius * 2];//@info: when array size > 50, nvcc always put array in local memory
//#else
    __shared__ float pixels_smem[CTA_size][CTA_size + filter_radius * 2 + 1];
//    float (&pixels)[CTA_size + filter_radius * 2] = *reinterpret_cast<float (*)[CTA_size + filter_radius * 2]>(&pixels_smem[thrd_id][0]);
//#endif
    float (&pixels)[CTA_size + filter_radius * 2] = keep_pixels_in_register ? pixels_register : *reinterpret_cast<float (*)[CTA_size + filter_radius * 2]>(&pixels_smem[thrd_id][0]);
//    for(int i = -2; i < 0; i ++)
//        conv_tile<filter_radius, blur_only, DoG_only, true>(cta_id, thrd_id, tex, img_size, img_size_inv, i, conv, pixels, buffer, pixels_last, DoG, blurred);
    for(int i = -div_ceil(preload_next_tile + filter_radius, CTA_size); i < tiles_per_row; i++){
        if(i < 0)
            conv_tile<filter_radius, blur_only, DoG_only, preload_next_tile, true>(cta_id, thrd_id, tex, img_size, img_size_inv, i, conv, pixels, buffer, pixels_last, DoG, blurred);
        else
            conv_tile<filter_radius, blur_only, DoG_only, preload_next_tile, false>(cta_id, thrd_id, tex, img_size, img_size_inv, i, conv, pixels, buffer, pixels_last, DoG, blurred);
    }
}

template <bool blur_only, bool DoG_only>
cudaError_t cuda_DoG_impl(const cudaTextureObject_t tex, const int2 img_size,
                     const std::vector<float>& filter1d,//left half and centre. symmetric filter assumed
                     const pitched_ptr<float>& DoG, const pitched_ptr<float>& blurred,
                     cudaStream_t stream)
{
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    //due to strange compiler issue, filter_radius == 8 is faster than 6 and 7, so we use 8 instead when we need 6 or 7
    if(filter1d.size() == 13 || filter1d.size() == 15){//@todo: maybe we do the same for all filter_radius < 8, to reduce binary size
        std::vector<float> filter1d_17(17, 0);
        std::copy(filter1d.begin(), filter1d.end(), filter1d_17.begin() + (17 - filter1d.size()) / 2);
        return cuda_DoG_impl<blur_only, DoG_only>(tex, img_size, filter1d_17, DoG, blurred, stream);
    }

    const int filter_size = (int)filter1d.size();
    if(filter1d.size() % 2 != 1)
        throw std::runtime_error("filter size must be odd number");
#define cuda_DoG_case(radius)\
    case radius:\
    {\
        std::array<float, (radius) * 2 + 1> arg_filter;\
        std::copy_n(filter1d.begin(), (radius) * 2 + 1, arg_filter.begin());\
        /*@todo: nvcc can't put everything in register when radius == 6,7,10. When NVCC is updated, recheck local memory usage against these magic numbers*/\
        constexpr bool keep_pixels_in_register = ((radius) <= 5 || (radius) == 8 || (radius) == 9);\
        kernel_DoG<(radius), blur_only, DoG_only, keep_pixels_in_register><<<div_ceil(img_size.y, CTA_size), CTA_size, 0, stream>>>(tex, img_size, float2{1.f/img_size.x, 1.f/img_size.y}, arg_filter, DoG, blurred);\
        break;\
    }
//        kernel_DoG<radius, blur_only, DoG_only, keep_pixels_in_register><<<div_ceil(img_size.y, CTA_size), CTA_size, 0, stream>>>(tex, img_size, float2{1.f/img_size.x, 1.f/img_size.y}, arg_filter, DoG, blurred);

    switch((filter_size - 1) / 2){
        cuda_DoG_case(1)
        cuda_DoG_case(2)
        cuda_DoG_case(3)
        cuda_DoG_case(4)
        cuda_DoG_case(5)
//        cuda_DoG_case(6)
//        cuda_DoG_case(7)
        cuda_DoG_case(8)
        cuda_DoG_case(9)
        cuda_DoG_case(10)
        cuda_DoG_case(11)
        cuda_DoG_case(12)
        cuda_DoG_case(13)
        cuda_DoG_case(14)
        cuda_DoG_case(15)
        cuda_DoG_case(16)
        default:
            throw std::runtime_error("unsupported filter radius");
    }
#undef cuda_DoG_case
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    return cudaGetLastError();
}

cudaError_t cuda_DoG(const cudaTextureObject_t tex, const int2 img_size,
                     const std::vector<float>& filter1d,
                     const pitched_ptr<float>& DoG,
                     const pitched_ptr<float>& blurred,
                     const bool DoG_only,
                     cudaStream_t stream){
    constexpr bool blur_only = false;
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    if(DoG_only)
        return cuda_DoG_impl<blur_only, true>(tex, img_size, filter1d, DoG, blurred, stream);
    else
        return cuda_DoG_impl<blur_only, false>(tex, img_size, filter1d, DoG, blurred, stream);
}

cudaError_t cuda_blur(const cudaTextureObject_t tex,
                      const int2 img_size,
                      const std::vector<float>& filter1d,
                      const pitched_ptr<float>& blurred,
                      cudaStream_t stream){
    constexpr bool blur_only = true;
#ifndef NDEBUG
    checkCudaError(cudaDeviceSynchronize());
#endif
    return cuda_DoG_impl<blur_only, false>(tex, img_size, filter1d, {nullptr, 0}, blurred, stream);
//    return cudaGetLastError();(void)tex; (void)(img_size); (void)filter1d;(void)blurred;(void)stream;
}

template <int thrdIters>
__global__ void kernel_down_sample(
        const pitched_ptr<float> dst, const uint2 dst_size,
        const pitched_ptr<const float> src)
{
    const unsigned x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned y = thrdIters * (blockDim.y * blockIdx.y + threadIdx.y);
    if (x >= dst_size.x || y >= dst_size.y)
        return;
    assert(src.pitch%4 == 0 && dst.pitch%4 == 0);
    const float* sPtr = src.ptr + src.pitch/sizeof(float)*2 * y;
    float* dPtr = dst.ptr + dst.pitch/sizeof(float) * y;

#if 0
    for(int n = 0; n < thrdIters; n++){
        const float2 data[2] = {
                __ldg((const float2*)&sPtr[x * 2]),
                __ldg((const float2*)&sPtr[src.pitch / sizeof(float)  + x * 2])
        };
        sPtr += src.pitch / sizeof(float)*2;
        dPtr[x] = 0.25f * (data[0].x + data[0].y + (data[1].x + data[1].y));
        dPtr += dst.pitch / sizeof(float);
    }
#else
    using texel = float2[2];
    constexpr int buffSize = 2;
    texel buffer[buffSize];

    for(int i = 0; i < 2; i++) {
        buffer[0][i] = __ldg((const float2*)&sPtr[src.pitch / sizeof(float) * i  + x * 2]);
    }
    sPtr += src.pitch/sizeof(float)*2;
    assert(thrdIters % buffSize == 0);
#pragma unroll(1)
    for (int m = 0; m < thrdIters/buffSize; m++) {
#pragma unroll
        for (int k = 0; k < buffSize; k++){
            const int n = m*buffSize+k;
            if (y + n >= dst_size.y)
                break;
            //prefetch
            if (y + n + 1 < dst_size.y) {
                for (int i = 0; i < 2; i++) {
                    buffer[(k + buffSize - 1) % buffSize][i] = __ldg((const float2 *) &sPtr[src.pitch / sizeof(float) * i + x * 2]);
                }
                sPtr += src.pitch / sizeof(float) * 2;
            }
            // @fixme: OpenCV uses nearest down sampling and seems the result is slightly better.
            dPtr[x] = 0.25f * ((buffer[k % buffSize][0].x + buffer[k % buffSize][0].y) +
                               (buffer[k % buffSize][1].x + buffer[k % buffSize][1].y));
            dPtr += dst.pitch / sizeof(float);
        }
    }
#endif
}

cudaError_t cuda_down_sample(
        const pitched_ptr<float>& dst, const uint2 dst_size,
        const pitched_ptr<const float>& src, cudaStream_t stream){
    constexpr int thrdIters = 32;
    dim3 dimBlock(128);
    dim3 dimGrid(div_ceil(dst_size.x, dimBlock.x), div_ceil(dst_size.y, dimBlock.y*32));
    kernel_down_sample<thrdIters><<<dimGrid, dimBlock, 0, stream>>>(dst, dst_size, src);
    return cudaGetLastError();
}
