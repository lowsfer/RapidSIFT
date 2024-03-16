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