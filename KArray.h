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
// Created by yao on 10/18/19.
//

#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime_api.h>

template <typename T, size_t... size>
struct KArray;

template <typename T, size_t size>
struct alignas(sizeof(T) * size < 16 ? sizeof(T) * size : 16) KArray<T, size>
{
    static constexpr uint32_t dimension = size;
    __host__ __device__ __forceinline__
    T& operator[](uint32_t idx) {return data[idx];}
    __host__ __device__ __forceinline__
    const T& operator[](uint32_t idx) const {return data[idx];}

    T* begin() {return &data[0];}
    T* end() {return &data[dimension];}

    T data[size];
};

template <typename T, size_t dim0, size_t... dims>
struct KArray<T, dim0, dims...> : public KArray<KArray<T, dims...>, dim0>{};

template <typename T, size_t... size>
bool operator!=(const KArray<T, size...>& a, const KArray<T, size...>& b){
    for (uint32_t i = 0; i < a.dimension; i++){
        if (a[i] != b[i]){
            return true;
        }
    }
    return false;
}

template <typename T, size_t... size>
bool operator==(const KArray<T, size...>& a, const KArray<T, size...>& b){
    return !(a != b);
}
