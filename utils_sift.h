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
// Created by yao on 8/10/17.
//

#pragma once
#include <type_traits>
#include <cmath>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <experimental/optional>
using std::experimental::optional;
using std::experimental::nullopt;

template<typename T>
constexpr T sqr(T x){
    return x * x;
}

template<typename T>
constexpr T div_ceil(T a, T b)
{
    static_assert(std::is_integral<T>::value, "typename T is not an integral type");
    return (a + b - 1) / b;
}

template<typename T>
constexpr T round_up(T a, T b)
{
    static_assert(std::is_integral<T>::value, "typename T is not an integral type");
    return (a + b - 1) / b * b;
}

template<typename T>
constexpr T round_down(T a, T b)
{
    static_assert(std::is_integral<T>::value, "typename T is not an integral type");
    return a / b * b;
}

//check if x is in range [lb, ub)
template<typename T>
constexpr bool in_range(T x, T lb, T ub){
    return x >= lb && x < ub;
}

__host__ __device__ __forceinline__ float fast_rcp(float x){
#ifdef __CUDACC__
    float result;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
#else
    return 1 / x;
#endif
}

__host__ __device__ __forceinline__ float fast_sqrt(float x){
#ifdef __CUDACC__
    float result;
    asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
#else
    return sqrtf(x);
#endif
}

__host__ __device__ __forceinline__ float fast_rsqrt(float x){
#ifdef __CUDACC__
    float result;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(x));
    return result;
#else
    return 1 / sqrtf(x);
#endif
}


class num_extremas_overflow : public std::runtime_error{
public:
    explicit num_extremas_overflow(const char* msg) : std::runtime_error(msg){}
};
class num_kpoints_overflow : public std::runtime_error{
public:
    explicit num_kpoints_overflow(const char* msg) : std::runtime_error(msg){}
};