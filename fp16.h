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
// Created by yao on 18/01/18.
//

#pragma once
#include <immintrin.h>
#include <cuda_fp16.h>

//template <typename FP16>
//float half2float(const FP16& val){
//    static_assert(sizeof(FP16) == 2, "fatal error");
//    __m128i src;
//    memcpy(&src, &val, sizeof(val));
//    const auto dst = _mm256_cvtph_ps(src);
//    float result;
//    memcpy(&result, &dst, sizeof(float));
//    return result;
//}
//
//template <typename FP16>
//const FP16 float2half(const float val){
//    static_assert(sizeof(FP16) == 2, "fatal error");
//    __m256 src;
//    memcpy(&src, &val, sizeof(val));
//    const auto dst = _mm256_cvtps_ph(src, 0);
//    FP16 result;
//    memcpy(&result, &dst, sizeof(FP16));
//    return result;
//}

//valgrind does not work with the intrinsics, use the cuda implementation instead
#define USE_CUDA_HALF_HOST_CONVERSION 0

#if USE_CUDA_HALF_HOST_CONVERSION || defined(__CUDACC__)
template <typename FP16>
inline float half2float(const FP16& val){
    static_assert(sizeof(FP16) == 2, "fatal error");
    half src;
    memcpy(&src, &val, 2);//respect strict aliasing rule
    return __half2float(src);
}

template <typename FP16>
inline const FP16 float2half(const float val){
    static_assert(sizeof(FP16) == 2, "fatal error");
    const half dst = __float2half(val);
    return reinterpret_cast<const FP16&>(dst);
}
#else
template <typename FP16>
inline float half2float(const FP16& val){
    static_assert(sizeof(FP16) == 2, "fatal error");
    unsigned short src;
    memcpy(&src, &val, 2);//respect strict aliasing rule
    return _cvtsh_ss(src); // valgrind does not work with this
}

template <typename FP16>
inline const FP16 float2half(const float val){
    static_assert(sizeof(FP16) == 2, "fatal error");
    const unsigned short dst = _cvtss_sh(val, 0); // valgrind does not work with this
    return reinterpret_cast<const FP16&>(dst);
//    FP16 result;
//    memcpy(&result, &dst, 2);//respect strict aliasing rule
//    return result;
}
#endif