//
// Created by yao on 12/01/18.
//

#pragma once
#include <experimental/optional>
#include <cuda_runtime_api.h>
#include <cassert>

class TexObj{
public:
    TexObj() = default;
    explicit TexObj(cudaTextureObject_t texture) : _tex(texture){}
    ~TexObj(){
        reset();
    }
    TexObj(const TexObj&) = delete;
    TexObj& operator=(const TexObj&) = delete;
    TexObj(TexObj&& other) noexcept :_tex{std::move(other._tex)}{
        other._tex = std::experimental::nullopt;
    }
    TexObj& operator=(TexObj&& other) noexcept {
        _tex = std::move(other._tex);
        other._tex = std::experimental::nullopt;
        return *this;
    }
    cudaTextureObject_t get() const {assert(_tex != std::experimental::nullopt); return *_tex;}
    void reset(){
        if(_tex != std::experimental::nullopt) {
//            printf("destroy texture %llu\n", *_tex);
            checkCudaError(cudaDestroyTextureObject(*_tex));
            _tex = std::experimental::nullopt;
        }
    }
    operator bool() const {return _tex != std::experimental::nullopt;}
private:
    std::experimental::optional<cudaTextureObject_t> _tex;
};