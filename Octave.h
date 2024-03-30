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

#pragma once
#include "Pitched2DTexture.h"

class Octave {
public:
    explicit Octave(size_t num_layers);

    ~Octave();

    Octave(Octave&& other) noexcept;

    Octave&operator=(Octave&& other) noexcept{
        _num_octave_layers = other._num_octave_layers;
        _img_size = other._img_size;
        _num_gauss_layers = other._num_gauss_layers;
        _gauss_layers = std::move(other._gauss_layers);
        _num_DoG_layers = other._num_DoG_layers;
        _DoG_layers = std::move(other._DoG_layers);
        other._num_DoG_layers = 0;
        other._img_size = {0,0};
        other._num_gauss_layers = 0;
        other._num_DoG_layers = 0;
        return *this;
    }

    void resize(size_t num_octave_layers);

    void build(const Pitched2DTexture<float>& input, Pitched2DTexture<float>& output, float sigma, cudaStream_t stream);

    const int2& img_size() const { return _img_size; }

    std::vector<cudaTextureObject_t> get_gauss_layer_textures(bool normalizedCoords = false) const;
    std::vector<cudaTextureObject_t> get_DoG_layer_textures() const;

    void dumpGaussLayers(char const* name, int idxGaussLayer, cudaStream_t stream);

private:
    size_t _num_octave_layers = 0;

    int2 _img_size = {0, 0};//updated by build()

    // _num_gauss_layers should be _num_octave_layers + 3, but the first and the last two layers are not used,
    // so I'm use _num_gauss_layer == _num_octave_layers to save some GPU RAM.
    size_t _num_gauss_layers = 0;//do not use _gauss_layer.size(), use _num_gauss_layers instead

    struct GaussLayer{
        Pitched2DTexture<float> data;//internal texture with normalized cooordinates
        TexObj tex_extra;//with unnormalized coordinates
    };
    std::vector<GaussLayer> _gauss_layers;//used for orientation calculation


    size_t _num_DoG_layers = 0;//_num_octave_layers + 2
    std::vector<Pitched2DTexture<float>> _DoG_layers;
};


