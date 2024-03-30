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
// Created by yao on 11/01/18.
//

#include <cuda_runtime.h>
#include <driver_types.h>
#include <cstring>
#include <texture_fetch_functions.h>
#include "utils_sift.h"
#include "/opt/cuda/include/texture_types.h"
#include <cstdint>
#include <cassert>

constexpr size_t width = 4;
constexpr size_t height = 4;

__global__ void kernel_texture_test(const cudaTextureObject_t tex, const float* __restrict__ ptr){
    for(int i = 0; i < int(width); i++){
        const float val = tex2D<float>(tex, 2.5f / 4, 0.5f);
        const float val2 = ptr[2];
        printf("%d: %f %f\n", i, val, val2);
    }
}

int main()
{
    void* ptr_dev;
    size_t pitch;
    checkCudaError(cudaMallocPitch(&ptr_dev, &pitch, sizeof(float) * width, height));
    assert(ptr_dev);

    const float data[height][width] = {
            0.2f, 0.4f, 0.6f, 0.8f,
            0.2f, 0.4f, 0.6f, 0.8f,
            0.2f, 0.4f, 0.6f, 0.8f,
            0.2f, 0.4f, 0.6f, 0.8f
    };
    checkCudaError(cudaMemcpy2D(ptr_dev, pitch, data, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice));
    checkCudaError(cudaDeviceSynchronize());

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceType::cudaResourceTypePitch2D;
    res_desc.res.pitch2D = {
            ptr_dev,
            cudaCreateChannelDesc<float>(),
            width,
            height,
            pitch
    };
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    //@fixme: mirror and warp mode are supported only with normalzed coordinate
    for(auto& e: tex_desc.addressMode) e = cudaTextureAddressMode::cudaAddressModeBorder;
    tex_desc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
    tex_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
    tex_desc.normalizedCoords = 1;
    cudaTextureObject_t tex;
    checkCudaError(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
    assert(tex != 0);

    for(int i = 0; i < 3; i++)
        kernel_texture_test<<<1,1,0,0>>>(tex, (const float*)ptr_dev);
    checkCudaError(cudaDeviceSynchronize());
    cudaFree(ptr_dev);

    return 0;
}