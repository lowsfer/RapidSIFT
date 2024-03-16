//
// Created by yao on 4/01/18.
//

#pragma once

#include <device_launch_parameters.h>
#include <cassert>
#include <cstdio>

template <int Radius>
struct conv1d{
    constexpr static int radius = Radius;
    constexpr static int filter_size = radius * 2 + 1;
    constexpr static float init_val = 0;

    __host__ __device__ __forceinline__ void init(const float (&val)[filter_size]){
#pragma unroll
        for (int n = 0; n < filter_size; n++) {
            accumulators[n] = init_val;
            for (int i = 0; i <= n; i++) {
                accumulators[i] += val[n] * filter(n - i);
            }
        }
    }

    __host__ __device__ __forceinline__ void set_filter(const float (&filter1d)[filter_size]){
#pragma unroll
        for(int i = 0; i < filter_size; i++)
            _filter[i] = filter1d[i];
    }

//    __device__ __forceinline__ void operator()(const float (&val)[filter_size], float(&out)[filter_size]){
//        for(int n = 0; n < filter_size; n++) {
//            out[n] = accumulators[n];
//            accumulators[n] = init_val;
//            for (int i = 0; i < filter_size; i++) {
//                const int idx = n - i;
//                accumulators[i] += val[n] * filter(idx < 0 ? idx + filter_size : idx);
//            }
//        }
//    }

    //count != filter_size is allowed only for the last iteration
    //the last call can have count = 0, but it must be called to fetch the last result. count = input_size % conv.filter_size can be 0 for the last call
    __host__ __device__ __forceinline__ void operator()(const float (&val)[filter_size], float(&out)[filter_size], int count = filter_size){
#if 0
        assert(count <= filter_size && count >= 0);
#pragma unroll
        for(int n = 0; n < filter_size; n++) {
            if(n >= count)
                break;
            out[n] = accumulators[n];
            accumulators[n] = init_val;
#pragma unroll
            for (int i = 0; i < filter_size; i++) {
                if (i > count)
                    break;
                const int idx = n - i;
                accumulators[i] += val[n] * filter(idx < 0 ? idx + filter_size : idx);
            }
        }
        if(count != filter_size)//This is the last iteration. Take the last result, which has been finished
            out[count] = accumulators[count];
#else
        if (count == filter_size)
            compute<true>(val, out, count);
        else
            compute<false>(val, out, count);
#endif
    }

    //count != filter_size is allowed only for the last iteration
    //the last call can have count = 0, but it must be called to fetch the last result. count = input_size % conv.filter_size can be 0 for the last call
    template <bool is_full = false>
    __host__ __device__ __forceinline__ void compute(const float (&val)[filter_size], float(&out)[filter_size], int count){
        assert(count <= filter_size && count >= 0);
        if(is_full)
            assert(count == filter_size);
#pragma unroll
        for(int n = 0; n < filter_size; n++) {
            if(!is_full && n >= count)
                break;
            out[n] = accumulators[n];
            accumulators[n] = init_val;
#pragma unroll
            for (int i = 0; i < filter_size; i++) {
                if (!is_full && i > count)
                    break;
                const int idx = n - i;
                accumulators[i] += val[n] * filter(idx < 0 ? idx + filter_size : idx);
            }
        }
        if(!is_full && count != filter_size)//This is the last iteration. Take the last result, which has been finished
            out[count] = accumulators[count];
    }

    template <int count>
    __host__ __device__ __forceinline__ void compute(const float* val /*length [count]*/, float* out /*length [std::min(filter_size, count + 1)]*/){
        static_assert(count <= filter_size && count >= 0);
#pragma unroll
        for(int n = 0; n < count; n++) {
            out[n] = accumulators[n];
            accumulators[n] = init_val;
#pragma unroll
            for (int i = 0; i <= count; i++) {
                const int idx = n - i;
                accumulators[i] += val[n] * filter(idx < 0 ? idx + filter_size : idx);
            }
        }
        if(count != filter_size)//This is the last iteration. Take the last result, which has been finished
            out[count] = accumulators[count];
    }

    __host__ __device__ __forceinline__ float filter(int idx){
        assert(idx >= 0 && idx < filter_size);
        return _filter[idx];
    }

    float _filter[filter_size];
    float accumulators[filter_size];
};



