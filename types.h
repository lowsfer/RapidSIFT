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
// Created by yao on 8/01/18.
//

#pragma once

#include "cuda_runtime_api.h"
#include <cstdint>

template<typename T>
struct pitched_ptr{
    using value_type = T;
    T* __restrict__ ptr;
    int pitch; // in bytes
    //i: y; j: x
    __host__ __device__ __forceinline__ constexpr T* get_ptr(int i, int j) const {
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(ptr) + pitch * i) + j;
    }
    __host__ __device__ __forceinline__ constexpr T* get_ptr(const int2& xy) const {
        return get_ptr(xy.y, xy.x);
    }
    __host__ __device__ __forceinline__ T& operator[](const int2& xy) const {
        return *get_ptr(xy);
    }

    template <bool Enabler = true, typename = typename std::enable_if<Enabler && !std::is_const<T>::value, void>::type>
    __host__ __device__ __forceinline__
    operator pitched_ptr<const T>() const{
        return pitched_ptr<const T>{ptr, pitch};
    }
};