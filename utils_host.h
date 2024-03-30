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
// Created by yao on 9/10/17.
//

#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cstdlib>
#include <new>
#include <vector>
#include <memory>
#include <random>
#include <string>
#include <algorithm>
#include <functional>

inline void checkCudaError(cudaError_t error){
    if(error != cudaSuccess) {
//        printf("cuda error: %d\n", error);
        throw std::runtime_error(cudaGetErrorName(error));
    }
}

template <class T>
struct cuda_host_allocator {
    typedef T value_type;
    cuda_host_allocator(){};
    template <class U>
    constexpr cuda_host_allocator(const cuda_host_allocator<U>& other) noexcept{}
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        cudaError_t error = cudaMallocHost(&ptr, n*sizeof(T));
        if(error != cudaSuccess)
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { checkCudaError(cudaFreeHost(ptr)); }
};
template <class T, class U>
bool operator==(const cuda_host_allocator<T>& a, const cuda_host_allocator<U>& b) { return true; }
template <class T, class U>
bool operator!=(const cuda_host_allocator<T>& a, const cuda_host_allocator<U>& b) { return false; }

template <class T>
struct cuda_managed_allocator {
    typedef T value_type;
    cuda_managed_allocator(){};
    template <class U>
    constexpr cuda_managed_allocator(const cuda_managed_allocator<U>& other) noexcept{}
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        cudaError_t error = cudaMallocManaged(&ptr, n*sizeof(T));
        if(error != cudaSuccess)
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { checkCudaError(cudaFree(ptr)); }
};
template <class T, class U>
bool operator==(const cuda_managed_allocator<T>& a, const cuda_managed_allocator<U>& b) { return true; }
template <class T, class U>
bool operator!=(const cuda_managed_allocator<T>& a, const cuda_managed_allocator<U>& b) { return false; }


template <class T>
struct cuda_device_allocator {
    typedef T value_type;
    cuda_device_allocator(){};
    template <class U>
    constexpr cuda_device_allocator(const cuda_device_allocator<U>& other) noexcept{}
    T* allocate(std::size_t n) {
        void* ptr = nullptr;
#ifdef NDEBUG
        cudaError_t error = cudaMalloc(&ptr, n*sizeof(T));
#else
        cudaError_t error = cudaMallocManaged(&ptr, n*sizeof(T));
#endif
        if(error != cudaSuccess)
            throw std::bad_alloc();
        return reinterpret_cast<T*>(ptr);
    }
    void deallocate(T* ptr, std::size_t) noexcept { checkCudaError(cudaFree(ptr)); }
};
template <class T, class U>
bool operator==(const cuda_device_allocator<T>& a, const cuda_device_allocator<U>& b) { return true; }
template <class T, class U>
bool operator!=(const cuda_device_allocator<T>& a, const cuda_device_allocator<U>& b) { return false; }

struct cuda_host_deleter {
    void operator()(void* p){
        checkCudaError(cudaFreeHost(p));
    }
};

struct cuda_device_deleter {
    void operator()(void* p){
        checkCudaError(cudaFree(p));
    }
};

struct cuda_array_deleter {
    void operator()(cudaArray_t p){
        checkCudaError(cudaFreeArray(p));
    }
};

inline void* host_alloc(size_t num_bytes){
    void* ptr;
    checkCudaError(cudaMallocHost(&ptr, num_bytes));
    return ptr;
}

inline void* device_alloc(size_t num_bytes){
    void* ptr;
    checkCudaError(cudaMalloc(&ptr, num_bytes));
    return ptr;
}

inline void* managed_alloc(size_t num_bytes){
    void* ptr;
    checkCudaError(cudaMallocManaged(&ptr, num_bytes));
    return ptr;
}

inline cudaStream_t make_cuda_stream(){
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream));
    return stream;
}

struct cuda_stream_deleter {
    void operator()(cudaStream_t s){
        checkCudaError(cudaStreamDestroy(s));
    }
};

inline cudaEvent_t make_cuda_event(uint32_t flag = cudaEventBlockingSync | cudaEventDisableTiming){
    cudaEvent_t event;
    checkCudaError(cudaEventCreateWithFlags(&event, flag));
    return event;
}

struct cuda_event_deleter {
    void operator()(cudaEvent_t e){
        cudaEventDestroy(e);
    }
};

void CUDART_CB callback_functor(cudaStream_t stream, cudaError_t status, void *data);

template<typename Func>
void stream_add_callback(cudaStream_t stream, Func&& func){
    auto callback = new std::function<void()>{std::forward<Func>(func)};
    const cudaError_t err = cudaStreamAddCallback(stream, callback_functor, callback, 0);
    if(err != cudaSuccess)
        delete callback;
    checkCudaError(err);
};

std::random_device::result_type get_true_random_number();

inline void require(bool val){
    if(!val)
        throw std::runtime_error("assertion failure");
}

template<typename T>
void copy_pitched(const T* src, size_t pitch_src, T* dst, size_t pitch_dst, size_t width, size_t height){
    for(size_t i = 0; i < height; i++){
        std::copy_n(src, width, dst);
        src = reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(src) + pitch_src);
        dst = reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(dst) + pitch_dst);
    }
}