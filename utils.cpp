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