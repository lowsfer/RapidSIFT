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
// Created by yao on 12/01/18.
//

#include <iostream>
#include "Octave.h"
#include "kernels.h"
#include <mutex>

Octave::Octave(size_t num_layers){
    resize(num_layers);
}

Octave::~Octave() {
}

void Octave::resize(size_t num_octave_layers){
    if(num_octave_layers == _num_octave_layers)
        return;
    _num_octave_layers = num_octave_layers;
    _num_DoG_layers = _num_octave_layers + 2;
    _num_gauss_layers = _num_octave_layers + 1;//should be _num_octave_layers + 3 but we drop the first one and the last one

    if(_gauss_layers.size() < _num_gauss_layers)
        _gauss_layers.resize(_num_gauss_layers);

    const size_t _num_DoG_layers = num_octave_layers + 2;
    _DoG_layers.reserve(_num_DoG_layers);
    static const cudaTextureDesc tex_desc_DoG = [](){
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        //mirror and warp mode are supported only with normalzied coordinate
        for(auto& e: tex_desc.addressMode)
            e = cudaTextureAddressMode::cudaAddressModeClamp;
        tex_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
        tex_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;
        return tex_desc;
    }();
    while(_DoG_layers.size() < _num_DoG_layers){
        _DoG_layers.emplace_back(tex_desc_DoG);
    }

}

void Octave::build(const Pitched2DTexture<float>& input, Pitched2DTexture<float>& output,
                   float sigma, cudaStream_t stream) {
    require(_num_octave_layers > 0);

    _img_size = {input.width(), input.height()};

    output.resize(input.width() / 2, input.height() / 2);

    static const cudaTextureDesc tex_desc_extra = [](){
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        //mirror and warp mode are supported only with normalzied coordinate
        for(auto& e: tex_desc.addressMode)
            e = cudaTextureAddressMode::cudaAddressModeClamp;
        tex_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
        tex_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
        tex_desc.normalizedCoords = 0;
        return tex_desc;
    }();

    for(unsigned n = 0; n < _num_gauss_layers; n++){
        const bool updated = _gauss_layers[n].data.resize(input.width(), input.height());
        if(updated || !_gauss_layers[n].tex_extra)
            _gauss_layers[n].tex_extra =_gauss_layers[n].data.create_texture(tex_desc_extra);
        assert(_gauss_layers[n].data.check_texture(_gauss_layers[n].tex_extra.get()));
    }
    for(unsigned n = 0; n < _num_DoG_layers; n++){
        _DoG_layers[n].resize(input.width(), input.height());
    }

    assert(_num_gauss_layers <= _gauss_layers.size());
    assert(_num_DoG_layers <= _DoG_layers.size());
    const float k = std::pow(2.f, 1.f / _num_octave_layers);
    for(unsigned n = 0; n < _num_DoG_layers; n++){
        const auto& in = [&]()->const Pitched2DTexture<float>&{
            if(n == 0)
                return input;
            else
                return _gauss_layers[n - 1].data;
        }();
        const Pitched2DTexture<float> dummy{};
        auto& out = [&]()->const Pitched2DTexture<float>&{
            if(n < _num_DoG_layers - 1)
                return _gauss_layers[n].data;
            else
                return dummy;
        }();
        const bool DoG_only = (n == _num_DoG_layers - 1);
        auto& DoG = _DoG_layers[n];
#if 0
        static std::once_flag warnOnce{};
        std::call_once(warnOnce, [](){
            printf("Using wrong sigma for SIFT, which happens to give signigicantly better match ratio");
        });
        // This is wrong but gives significantly better correct match ratio!
        const float sigma_total = sigma * std::pow(k, n);
        const float sigma_prev = (n == 0 ? 0.f : sigma * std::pow(k, n - 1));
#else
        const float sigma_prev = std::pow(k, n)*sigma;
        const float sigma_total = sigma_prev * k;
#endif
        const float sigma_delta = std::sqrt(sqr(sigma_total) - sqr(sigma_prev));
        auto filter1d = make_gaussian_filter(sigma_delta);
        assert(in.check_texture(in.get_tex()));
        checkCudaError(cuda_DoG(in.get_tex(), int2{in.width(), in.height()}, filter1d, DoG.get_ptr(), DoG_only?pitched_ptr<float>{nullptr, 0}:out.get_ptr(), DoG_only, stream));
    }
    checkCudaError(cuda_down_sample(output.get_ptr(), uint2{(unsigned)output.width(), (unsigned)output.height()}, _gauss_layers[_num_octave_layers-1].data.get_ptr(), stream));
}

Octave::Octave(Octave &&other) noexcept
        : _num_octave_layers{other._num_octave_layers},
          _num_gauss_layers{other._num_gauss_layers},
          _gauss_layers{std::move(other._gauss_layers)},
          _num_DoG_layers{other._num_DoG_layers},
          _DoG_layers{std::move(other._DoG_layers)}{
    other._num_octave_layers = 0;
    other._num_gauss_layers = 0;
    other._num_DoG_layers = 0;
}

std::vector<cudaTextureObject_t> Octave::get_gauss_layer_textures(const bool normalizedCoords) const {
    std::vector<cudaTextureObject_t> result(_num_gauss_layers);
    std::transform(_gauss_layers.begin(), _gauss_layers.begin() + _num_gauss_layers, result.begin(),
                   [normalizedCoords](const GaussLayer& layer){
                       assert(bool(layer.tex_extra));
                       return normalizedCoords ? layer.data.get_tex() : layer.tex_extra.get();
                   });
    return result;
}

std::vector<cudaTextureObject_t> Octave::get_DoG_layer_textures() const {
    std::vector<cudaTextureObject_t> result(_num_DoG_layers);
    std::transform(_DoG_layers.begin(), _DoG_layers.begin() + _num_DoG_layers, result.begin(),
                   [](const Pitched2DTexture<float>& layer){
                       return layer.get_tex();
                   });
    return result;
}

#include <fstream>

void Octave::dumpGaussLayers(char const* name, int idxGaussLayer, cudaStream_t stream)
{
    std::vector<float> data(img_size().x * img_size().y);
    auto const& layer = _gauss_layers.at(idxGaussLayer);
    checkCudaError(cudaMemcpy2DAsync(data.data(), sizeof(float)*img_size().x, layer.data.get_ptr().ptr, layer.data.get_ptr().pitch, sizeof(float) * img_size().x, img_size().y, cudaMemcpyDeviceToHost, stream));
    checkCudaError(cudaStreamSynchronize(stream));
    std::ofstream fout(name, std::ios::trunc | std::ios::binary);
    fout.write((char const*)data.data(), sizeof(float) * data.size());
}
