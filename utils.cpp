//
// Created by yao on 18/01/18.
//
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include "utils_host.h"

void CUDART_CB callback_functor(cudaStream_t stream, cudaError_t status, void *data){
    (void)(stream);
    std::unique_ptr<std::function<void()>> func{reinterpret_cast<std::function<void()>*>(data)};
    checkCudaError(status);
    nvtxRangePush("callback");
    (*func)();
    nvtxRangePop();
}