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
// Created by yao on 11/10/17.
//

#pragma once

#include "types.h"
#include "utils_sift.h"
#include <cassert>
#include <cstring>
#include "utils_host.h"
#include "TexObj.h"
#include <cuda_runtime.h>
#include <driver_types.h>

template<typename T>
class Pitched2DTexture {
public:
    typedef T elem_type;
    constexpr static size_t alignment = 512;
public:
    Pitched2DTexture() = default;

    explicit Pitched2DTexture(cudaTextureDesc tex_desc)
            :_tex_desc{tex_desc}{}

    Pitched2DTexture(Pitched2DTexture<T>&& other) noexcept
            :_data{std::move(other._data)}, _pitch(other._pitch),
             _bytes{other._bytes},
             _tex_desc{other._tex_desc},
             _texobj{std::move(other._texobj)},
             _width{other._width}, _height{other._height} {
        other._pitch = 0;
        other._bytes = 0;
        other._width = 0;
        other._height = 0;
    }

    ~Pitched2DTexture() = default;

    void reset(){
        _height = 0;
        _width = 0;
        _texobj.reset();
        _bytes = 0;
        _pitch = 0;
        _data.reset();
    }

    pitched_ptr<T> get_ptr() const{
        assert(_data.get() != nullptr);
        return pitched_ptr<T>{_data.get(), (int)_pitch};
    }

    cudaTextureObject_t get_tex() const{
        assert(_texobj);
        return _texobj.get();
    }

    int width() const {return _width;}
    int height() const {return _height;}

    //if existing array is larger than (width, height), no allocation will happen
    bool resize(int width, int height){
        if(width == _width && height == _height)
            return false;

        _width = width;
        _height = height;
        _texobj.reset();

//        printf("update tex size ");
        {//allocate memory if necessary
            const size_t pitch_new = round_up(sizeof(elem_type) * width, alignment);
            const size_t bytes_required = pitch_new * height;
            if (_data == nullptr || _bytes < bytes_required) {
                void *ptr = device_alloc(bytes_required);
                _data.reset(reinterpret_cast<T *>(ptr));
                _bytes = bytes_required;
//                printf("with allocation\n");
            }else {
//                printf("without allocation\n");
            }
            _pitch = pitch_new;
        }

        update_texture();
        return true;
    }

    const cudaTextureDesc& get_tex_desc() const{
        return _tex_desc;
    }

    void set_tex_desc(const cudaTextureDesc& tex_desc){
        _tex_desc = tex_desc;
        if(_data)
            update_texture();
    }

    void update_texture(){
        _texobj = create_texture(_tex_desc);
//        printf("created texture %llu\n", _texobj.get());
    }

    //check if the texture's resource is from this object
    bool check_texture(cudaTextureObject_t tex) const{
        cudaResourceDesc res_desc;
        checkCudaError(cudaGetTextureObjectResourceDesc(&res_desc, tex));
        return res_desc.resType == cudaResourceType::cudaResourceTypePitch2D
               && res_desc.res.pitch2D.devPtr == _data.get()
               && res_desc.res.pitch2D.height == (unsigned)_height
               && res_desc.res.pitch2D.width == (unsigned)_width
               && res_desc.res.pitch2D.pitchInBytes == _pitch;
        //@todo: currently not checking channel desc.
    }

    //this call does not update internal texture
    TexObj create_texture(const cudaTextureDesc& tex_desc) const{
        require(_data != nullptr);
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceType::cudaResourceTypePitch2D;
        res_desc.res.pitch2D = {
                _data.get(),
                cudaCreateChannelDesc<T>(),
                (size_t)_width,
                (size_t)_height,
                _pitch
        };
        cudaTextureObject_t tex = 0;
        checkCudaError(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
        return TexObj(tex);
    }

private:
    std::unique_ptr<T, cuda_device_deleter> _data = nullptr;
    size_t _pitch = 0;
    size_t _bytes = 0;//number of allocated bytes
    cudaTextureDesc _tex_desc = [](){
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        //mirror and warp mode are supported only with normalzied coordinate
        for(auto& e: tex_desc.addressMode)
            e = cudaTextureAddressMode::cudaAddressModeMirror;
        tex_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
        tex_desc.readMode = (std::is_integral<T>::value ? cudaTextureReadMode::cudaReadModeNormalizedFloat : cudaTextureReadMode::cudaReadModeElementType);
        tex_desc.normalizedCoords = 1;
        return tex_desc;
    }();
    TexObj _texobj;
    int _width = 0;
    int _height = 0;
};


