//
// Created by yao on 7/12/17.
//

#pragma once
#include <functional>
#include <cuda_runtime_api.h>
#include "utils_host.h"
#include <mutex>
#include <cstring>
#include <nvToolsExt.h>

class cuda_async_copy_engine{
public:
    explicit cuda_async_copy_engine(cudaStream_t stream)
            :_buffer(16<<20), _stream(stream), _event(make_cuda_event()){}

    ~cuda_async_copy_engine(){
        cudaEventSynchronize(_event.get());
    }

    cuda_async_copy_engine(cuda_async_copy_engine&& src) noexcept
            :_buffer{std::move(src._buffer)}, _stream{src._stream},
             _event{std::move(src._event)}
    {}

    cuda_async_copy_engine& operator=(cuda_async_copy_engine&& src) noexcept
    {
        _buffer = std::move(src._buffer);
        _stream = src._stream;
        _event = std::move(src._event);
        return *this;
    }
    
//    using src_getter_t = std::function<const void*()>;
//    using dst_getter_t = std::function<void*()>;
    using callback_dev2host_finish_t = std::function<void()>;

    cudaStream_t get_stream() const{
        return _stream;
    }

    void rebind_stream(cudaStream_t stream_new){
        if(stream_new != _stream) {
            printf("migrating copy engine ...\n");
            checkCudaError(cudaStreamWaitEvent(stream_new, _event.get(), 0));
            _stream = stream_new;
        }
    }

    template<typename src_getter_t>
    void host2dev_pitched(void* dst, size_t pitch_dst, src_getter_t&& src_getter, size_t pitch_src, size_t width, size_t height){
        static_assert(std::is_convertible<std::result_of_t<src_getter_t()>, const void*>::value, "wrong signature for src_getter");
        std::lock_guard<std::mutex> lk(_mutex);

        const size_t num_bytes_src = pitch_src * height;
        prepare(num_bytes_src);

        host2buffer(std::forward<src_getter_t>(src_getter), num_bytes_src);
        checkCudaError(cudaMemcpy2DAsync(dst, pitch_dst, _buffer.data(), pitch_src, width, height, cudaMemcpyHostToDevice, _stream));

        finish();
    }

    template<typename src_getter_t>
    void host2dev(void* dst, src_getter_t&& src_getter, size_t num_bytes){
        static_assert(std::is_convertible<std::result_of_t<src_getter_t()>, const void*>::value, "wrong signature for src_getter");
        host2dev_pitched(dst, num_bytes, std::forward<src_getter_t>(src_getter), num_bytes, num_bytes, 1);
    }

    template<typename src_getter_t>
    void host2array_pitched(cudaArray_t dst, size_t wOffset, size_t hOffset,
                    src_getter_t&& src_getter, size_t pitch_src, size_t width, size_t height){
        static_assert(std::is_convertible<std::result_of_t<src_getter_t()>, const void*>::value, "wrong signature for src_getter");
        std::lock_guard<std::mutex> lk(_mutex);

        const size_t num_bytes_src = pitch_src * height;
        prepare(num_bytes_src);

        host2buffer(std::forward<src_getter_t>(src_getter), num_bytes_src);
        checkCudaError(cudaMemcpy2DToArrayAsync(dst, wOffset, hOffset, _buffer.data(), pitch_src, width, height, cudaMemcpyHostToDevice, _stream));

        finish();
    }

    template<typename src_getter_t>
    void host2array(cudaArray_t dst, src_getter_t&& src_getter, size_t width, size_t height){
        static_assert(std::is_convertible<std::result_of_t<src_getter_t()>, const void*>::value, "wrong signature for src_getter");
        host2array_pitched(dst, 0, 0, std::forward<src_getter_t>(src_getter), width, width, height);
    }

    template<typename dst_getter_t>
    void dev2host_pitched(dst_getter_t&& dst_getter, size_t pitch_dst, void* src, size_t pitched_src, size_t width, size_t height,
                          callback_dev2host_finish_t&& on_finish = nullptr){
        static_assert(std::is_convertible<std::result_of_t<dst_getter_t()>, void*>::value, "wrong signature for dst_getter");
        std::lock_guard<std::mutex> lk(_mutex);

        const size_t num_bytes_dst = pitch_dst * height;
        prepare(num_bytes_dst);

        checkCudaError(cudaMemcpy2DAsync(_buffer.data(), pitch_dst, src, pitched_src, width, height, cudaMemcpyDeviceToHost, _stream));
        buffer2host(std::forward<dst_getter_t>(dst_getter), num_bytes_dst, std::forward<callback_dev2host_finish_t>(on_finish));

        finish();
    }

    template<typename dst_getter_t>
    void dev2host(dst_getter_t&& dst_getter, void* src, size_t num_bytes,
                  callback_dev2host_finish_t&& on_finish = nullptr){
        static_assert(std::is_convertible<std::result_of_t<dst_getter_t()>, void*>::value, "wrong signature for dst_getter");
        dev2host_pitched(std::forward<dst_getter_t>(dst_getter), num_bytes, src, num_bytes, num_bytes, 1,
                         std::forward<callback_dev2host_finish_t>(on_finish));
    }

    template<typename dst_getter_t>
    void array2host_pitched(dst_getter_t&& dst_getter, size_t pitch_dst, cudaArray_t src, size_t wOffset,
                            size_t hOffset, size_t width, size_t height,
                            callback_dev2host_finish_t&& on_finish = nullptr){
        static_assert(std::is_convertible<std::result_of_t<dst_getter_t()>, void*>::value, "wrong signature for dst_getter");
        std::lock_guard<std::mutex> lk(_mutex);

        const size_t num_bytes_dst = pitch_dst * height;
        prepare(num_bytes_dst);

        checkCudaError(cudaMemcpy2DFromArrayAsync(_buffer.data(), pitch_dst, src, wOffset, hOffset, width, height, cudaMemcpyDeviceToHost, _stream));
        buffer2host(std::forward<dst_getter_t>(dst_getter), num_bytes_dst, std::forward<callback_dev2host_finish_t>(on_finish));

        finish();
    }

    template<typename dst_getter_t>
    void array2host(dst_getter_t&& dst_getter, cudaArray_t src, size_t width, size_t height,
                    callback_dev2host_finish_t&& on_finish = nullptr){
        static_assert(std::is_convertible<std::result_of_t<dst_getter_t()>, void*>::value, "wrong signature for dst_getter");
        array2host_pitched(std::forward<dst_getter_t>(dst_getter), width, src, 0, 0, width, height,
                           std::forward<callback_dev2host_finish_t>(on_finish));
    }

    void sync(){
        checkCudaError(cudaEventSynchronize(_event.get()));
    }

private:
    void prepare(size_t buffer_size_required){
        if(_buffer.size() < buffer_size_required) {
//            printf("buffer resizing ...\n");
            sync();
            _buffer.resize(buffer_size_required * 2);
        }
    }

    void finish(){
        checkCudaError(cudaEventRecord(_event.get(), _stream));
    }

    template<typename src_getter_t>
    void host2buffer(src_getter_t&& src_getter, size_t num_bytes_src){
        static_assert(std::is_convertible<std::result_of_t<src_getter_t()>, const void*>::value, "wrong signature for src_getter");
        uint8_t* buffer = _buffer.data();
        auto host_memcpy = [src_getter{std::forward<src_getter_t >(src_getter)}, buffer, num_bytes_src](){
            nvtxRangePush("memcpy paged to pinned");
            memcpy(buffer, src_getter(), num_bytes_src);
            nvtxRangePop();
        };
        stream_add_callback(_stream, host_memcpy);
    }

    template<typename dst_getter_t>
    void buffer2host(dst_getter_t&& dst_getter, size_t num_bytes_dst, callback_dev2host_finish_t&& on_finish){
        static_assert(std::is_convertible<std::result_of_t<dst_getter_t()>, void*>::value, "wrong signature for dst_getter");
        uint8_t* buffer = _buffer.data();
        auto host_memcpy = [dst_getter{std::forward<dst_getter_t>(dst_getter)}, buffer, num_bytes_dst, on_finish{std::forward<callback_dev2host_finish_t>(on_finish)}](){
            nvtxRangePush("memcpy pinned to paged");
            memcpy(dst_getter(), buffer, num_bytes_dst);
            nvtxRangePop();
            if(on_finish != nullptr) {
                nvtxRangePush("after memcpy pinned to paged");
                on_finish();
                nvtxRangePop();
            }
        };
        stream_add_callback(_stream, host_memcpy);
    }

private:
    std::mutex _mutex;
    std::vector<uint8_t, cuda_host_allocator<uint8_t>> _buffer;
    cudaStream_t _stream;
    std::unique_ptr<CUevent_st, cuda_event_deleter> _event;
};