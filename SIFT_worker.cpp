//
// Created by yao on 13/01/18.
//

#include "SIFT_worker.h"
#include "kernels.h"

const static size_t init_idx_input = 0;
const static float init_sigma = SIFT_INIT_SIGMA;
const static size_t init_max_extremas = ((8 << 20) - sizeof(GPUArray<ScaleSpaceExtrema>)) / sizeof(ScaleSpaceExtrema);
const static size_t init_max_kpoints = ((2 << 20) - sizeof(GPUArray<KeyPoint>)) / sizeof(KeyPoint);
const static size_t max_num_octaves = 8;

// seems larger patch, e.g. 9.5f, gives more inliers for two-view matching and oblique multi-viuew, but turns out to be bad for multi-view of planar scene.
// @todo: Maybe because large patch causes wrong matches with neighbour key points. Need more investigation.
const static float sosnetMagFactor = 4.f;
// const static float sosnetMagFactor = 9.5f;

SIFT_worker::SIFT_worker(DescType descType) : mDescType{descType}, _thread_pool(1) {
    auto tex_desc_dev_input = [](){
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        //mirror and warp mode are supported only with normalzied coordinate
        for(auto& e: tex_desc.addressMode)
            e = cudaTextureAddressMode::cudaAddressModeClamp;
        tex_desc.filterMode = cudaTextureFilterMode::cudaFilterModeLinear;
        tex_desc.readMode = cudaTextureReadMode::cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;
        return tex_desc;
    }();
    _dev_input.set_tex_desc(tex_desc_dev_input);
    reserve_extremas(init_max_extremas);
    reserve_kpoints(init_max_kpoints);
    int device;
    checkCudaError(cudaGetDevice(&device));
    _thread_pool.enqueue(
            [device](){
                checkCudaError(cudaSetDevice(device));
            }
    );
    if (descType == DescType::kSOSNet) {
        _sosnet = createSOSNet();
    }
}

SIFT_worker::~SIFT_worker() {
    syncStream();
}

void SIFT_worker::feed(std::function<const void*()>&& src_getter, int width, int height, float thres_contrast, bool up_sample){
    preprocess(std::move(src_getter), width, height, up_sample);
    build_DoG_layers(up_sample);
    find_keypoints(thres_contrast);
    assign_orientation(up_sample);
    describe(up_sample);
}

void SIFT_worker::feed(std::function<const void*()>&& src_getter, int width, int height,
        uint32_t targetNbKPoints, float minOverDetectRatio, float init_thres_contrast, bool up_sample,
        float min_thres_contrast){
    float thres_contrast = init_thres_contrast;
    preprocess(std::move(src_getter), width, height, up_sample);
    build_DoG_layers(up_sample);
    const uint32_t minNumDetectedKPoints {uint32_t(targetNbKPoints * minOverDetectRatio)};
    do{
        find_keypoints(thres_contrast);
        if (_num_kpoints < minNumDetectedKPoints){
            if (thres_contrast <= min_thres_contrast) // must break on equal
            {
                break;
            }
            thres_contrast = std::min(_predicated_thres_contrast, std::min(min_thres_contrast, thres_contrast * 0.5f));
        }
    }while (_num_kpoints < minNumDetectedKPoints);
    _predicated_thres_contrast = thres_contrast;
    if (targetNbKPoints < _num_kpoints){
        _num_kpoints = _anms_filter.filterDevKPointsSync(_kpoints.get(), targetNbKPoints, 1.11f, true);
    }
    assign_orientation(up_sample);
    describe(up_sample);
}

void SIFT_worker::preprocess(std::function<const void *()> &&src_getter, const int width, const int height, bool up_sample) {
    require(width > 0 && height > 0);
    _dev_input.resize(width, height);
    _copy_engine.host2dev_pitched(
            _dev_input.get_ptr().ptr, (size_t)_dev_input.get_ptr().pitch,
            std::move(src_getter), sizeof(uint8_t) * width, sizeof(uint8_t) * width, (size_t)height);
    //up_sample and blur, then store data into _workspace[init_idx_input]
    const int2 img_size_fake = up_sample ? int2{width * 2, height * 2} : int2{width, height};
    const float sigma = std::sqrt(sqr(_sigma) - sqr(init_sigma) * (up_sample ? 4.f : 1.f));
    const auto filter1d = make_gaussian_filter(std::max(sigma, 0.1f));
    _workspace[init_idx_input].resize(img_size_fake.x, img_size_fake.y);
    checkCudaError(cuda_blur(_dev_input.get_tex(), img_size_fake, filter1d, _workspace[init_idx_input].get_ptr(), _stream.get()));
    _img_size = img_size_fake;
}

void SIFT_worker::build_DoG_layers(bool up_sample) {
    //make sure existing octaves have enough layers
    for (unsigned i = 0; i < _num_octaves; i++)
        _octaves.at(i).resize(_num_octave_layers);

    //make sure we have enough octaves
    const int num_octaves = std::min(
            (int)max_num_octaves,
            (int)std::round(std::log((float)std::min(_img_size.x, _img_size.y)) / std::log(2.f) - 2) + (up_sample ? 1 : 0) - 2 //I see usually the last two octaves contain no extremas, so I reduce num_octaves by 2 here.
    );
    require(num_octaves > 0);
    while (_octaves.size() < (size_t) num_octaves) {
        _octaves.emplace_back(_num_octave_layers);
    }
    _num_octaves = (size_t) num_octaves;

    size_t idx_input = init_idx_input;
    for (unsigned n = 0; n < _num_octaves; n++) {
        const size_t idx_output = (idx_input + 1) % _workspace.size();
        auto &in = _workspace.at(idx_input);
        auto &out = _workspace.at(idx_output);
        _octaves.at(n).build(in, out, _sigma, _stream.get());
        idx_input = idx_output;
    }
}

void SIFT_worker::find_keypoints(float thres_contrast) {
    std::vector<cudaTextureObject_t> textures;
    textures.reserve((_num_octave_layers + 2) * _num_octaves + (_num_octave_layers + 1) * _num_octaves);
    for(size_t i = 0; i < _num_octaves; i++){
        const auto tmp = _octaves[i].get_DoG_layer_textures();
        textures.insert(textures.end(), tmp.begin(), tmp.end());
    }
    assert(textures.size() == (_num_octave_layers + 2) * _num_octaves);
    for(size_t i = 0; i < _num_octaves; i++){
        const auto tmp = _octaves[i].get_gauss_layer_textures(false);
        textures.insert(textures.end(), tmp.begin(), tmp.end());
    }
    assert(textures.size() == (_num_octave_layers + 2) * _num_octaves + (_num_octave_layers + 1) * _num_octaves);
    if(_capacity_textures < textures.size()) {
        _textures.reset(cuda_device_allocator<cudaTextureObject_t>{}.allocate(textures.size()));
        _capacity_textures = textures.size();
    }
    _copy_engine.host2dev(_textures.get(), [textures{std::move(textures)}](){return textures.data();}, sizeof(textures[0]) * textures.size());
    uint32_t num_kpoints = 0u;
    do {
        if(num_kpoints > _max_num_kpoints)
            reserve_kpoints(std::max(num_kpoints, _max_num_kpoints * 2));
        for (int i = 0; i < (int) _num_octaves; i++) {
            const auto img_size = _octaves[i].img_size();
            uint32_t num_extremas = 0u;
            do {
                if (num_extremas > _max_num_extremas)
                    reserve_extremas(std::max(num_extremas, _max_num_extremas * 2));
                checkCudaError(
                        cuda_find_extrema(&_textures[(_num_octave_layers + 2) * i], int(_num_octave_layers + 2), i,
                                          img_size,
                                          _extremas.get(), _max_num_extremas,
                                          0.5f * thres_contrast / _num_octave_layers,
                                          true, _stream.get()));
                syncStream();
                checkCudaError(cuda_get_num_extremas(_extremas.get(), &num_extremas, _stream.get()));
                syncStream();
            } while (num_extremas > _max_num_extremas);

            const bool reset_kpoints = (i == 0);
            checkCudaError(cuda_make_keypoints(
                    _extremas.get(), num_extremas, &_textures[(_num_octave_layers + 2) * i],
                    int(_num_octave_layers + 2), i, img_size,
                    int2{SIFT_IMG_BORDER, SIFT_IMG_BORDER},
                    int2{img_size.x - SIFT_IMG_BORDER, img_size.y - SIFT_IMG_BORDER},
                    thres_contrast, _thres_edge, _sigma,
                    _kpoints.get(), _max_num_kpoints, reset_kpoints, _stream.get()));
        }
        syncStream();
        checkCudaError(cuda_get_num_kpoints(_kpoints.get(), &num_kpoints, _stream.get()));
        syncStream();
    }while(num_kpoints > _max_num_kpoints);
    _num_kpoints = num_kpoints;
}

void SIFT_worker::assign_orientation(bool up_sample) {
    assert(_num_kpoints <= _max_num_kpoints);
    const uint32_t num_kpoints_before_split = _num_kpoints;
    uint32_t num_kpoints = _num_kpoints;
    do {
        if (num_kpoints > _max_num_kpoints) {
            auto backup = std::move(_kpoints);
            reserve_kpoints(std::max(num_kpoints, _max_num_kpoints * 2));
            checkCudaError(cudaMemcpyAsync(_kpoints.get(), backup.get(),
                    sizeof(*_kpoints) + sizeof(_kpoints->data[0]) * num_kpoints_before_split,
                    cudaMemcpyDeviceToDevice, _stream.get()));
        }
        // when _kpoints overflows, the first num_kpoints_before_split kpoints should be unchanged except their orientation member data.
        checkCudaError(cuda_assign_orientation(&_textures[(_num_octave_layers + 2) * _num_octaves],
                                               (int) _num_octave_layers, (int) _num_octaves, _octaves.at(0).img_size(), _kpoints.get(),
                                               _max_num_kpoints, num_kpoints, up_sample, _stream.get()));
        // note that due to split, orientation assignment may increase number of key points.
        checkCudaError(cuda_get_num_kpoints(_kpoints.get(), &num_kpoints, _stream.get()));
    }while (num_kpoints > _max_num_kpoints);
    _num_kpoints = num_kpoints;
}

void SIFT_worker::describe(bool up_sample) {
    assert(_num_kpoints <= _max_num_kpoints);
    if (mDescType == DescType::kSIFT || mDescType == DescType::kRootSIFT) {
        checkCudaError(cuda_describe(
            &_textures[(_num_octave_layers + 2) * _num_octaves], (int)_num_octave_layers, (int)_num_octaves, _octaves.at(0).img_size(), up_sample,
            _kpoints.get(), _num_kpoints, 64.f, _descriptors.get(), mDescType == DescType::kRootSIFT, _stream.get()));
        return;
    }
    assert(mDescType == DescType::kSOSNet);
    float const patchQuantScale = _sosnet->getInputScale();
    checkCudaError(cuda_makePatch(_dev_input.get_tex(), int2{_dev_input.width(), _dev_input.height()},
        _kpoints.get(), _num_kpoints, sosnetMagFactor, 1.f / patchQuantScale, _patches.get(), _stream.get()));
    for (int i = 0; i < (int)_num_kpoints; i += _sosnet->maxBatchSize) {
        int32_t const batch = std::min(_sosnet->maxBatchSize, (int32_t)_num_kpoints - i);
        _sosnet->infer((uint8_t(*)[128])&_descriptors[i], (int8_t(*)[32][32])&_patches[i], batch, _stream.get());
    }
}

std::vector<KeyPoint> SIFT_worker::get_keypoints() {
    assert([&]()->bool{
        uint32_t num_kpoints;
        checkCudaError(cuda_get_num_kpoints(_kpoints.get(), &num_kpoints, _stream.get()));
        return num_kpoints == _num_kpoints;
    }());
    std::vector<KeyPoint> result(_num_kpoints);
    syncStream();
    checkCudaError(cudaMemcpyAsync(result.data(), _kpoints->data, sizeof(KeyPoint) * _num_kpoints, cudaMemcpyDeviceToHost, _stream.get()));
    syncStream();
    return result;
}

std::vector<SiftDescriptor> SIFT_worker::get_descriptors() {
    assert([&]()->bool{
        uint32_t num_kpoints;
        checkCudaError(cuda_get_num_kpoints(_kpoints.get(), &num_kpoints, _stream.get()));
        return num_kpoints == _num_kpoints;
    }());
    std::vector<SiftDescriptor> result(_num_kpoints);
    syncStream();
    cudaMemcpyAsync(result.data(), _descriptors.get(), sizeof(SiftDescriptor) * _num_kpoints, cudaMemcpyDeviceToHost, _stream.get());
    syncStream();
    return result;
}

void SIFT_worker::reserve_extremas(uint32_t max_num_extremas) {
    _max_num_extremas = max_num_extremas;
    _extremas.reset(reinterpret_cast<GPUArray<ScaleSpaceExtrema>*>(cuda_device_allocator<uint8_t>{}.allocate(sizeof(GPUArray<ScaleSpaceExtrema>) + sizeof(ScaleSpaceExtrema) * _max_num_extremas)));
}

void SIFT_worker::reserve_kpoints(uint32_t max_num_kpoints) {
    _max_num_kpoints = max_num_kpoints;
    _kpoints.reset(reinterpret_cast<GPUArray<KeyPoint>*>(cuda_device_allocator<uint8_t>{}.allocate(sizeof(GPUArray<KeyPoint>) + sizeof(KeyPoint) * _max_num_kpoints)));
    _descriptors.reset(cuda_device_allocator<SiftDescriptor>{}.allocate(_max_num_kpoints));
    if (mDescType == DescType::kSOSNet) {
        _patches.reset(cuda_device_allocator<Patch32x32>{}.allocate(_max_num_kpoints));
    }
}

std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>>
SIFT_worker::detect_and_compute_async(std::function<const void *()> &&src_getter, int width, int height,
        float thres_contrast, bool up_sample) {
    auto task = [this, src_getter{std::move(src_getter)}, width, height, thres_contrast, up_sample]() mutable{
        std::lock_guard<std::mutex> lk{_mutex};
        feed(std::move(src_getter), width, height, thres_contrast, up_sample);
        std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>> result = std::make_pair(
                get_keypoints(), get_descriptors()
        );
        return result;
    };

    return _thread_pool.enqueue(std::move(task));
}

std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>>
SIFT_worker::uniform_detect_and_compute_async(std::function<const void *()> &&src_getter, int width, int height,
                                              uint32_t targetNbKPoints, float minOverDetectRatio,
                                              float init_thres_contrast, bool up_sample, float min_thres_contrast) {
    auto task = [this, src_getter{std::move(src_getter)}, width, height, targetNbKPoints, minOverDetectRatio, init_thres_contrast, up_sample, min_thres_contrast]() mutable {
        std::lock_guard<std::mutex> lk{_mutex};
        feed(std::move(src_getter), width, height, targetNbKPoints, minOverDetectRatio, init_thres_contrast, up_sample, min_thres_contrast);
        std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>> result = std::make_pair(
                get_keypoints(), get_descriptors()
        );
        return result;
    };
    return _thread_pool.enqueue(std::move(task));
}

std::future<std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>>>
    SIFT_worker::uniform_detect_compute_and_abstract_async(
            std::function<const void*()>&& src_getter, int width, int height,
            // The detector should detect at least targetNbKPoints * overDetectRatio key-points before adaptive-NMS
            uint32_t targetNbKPoints, float minOverDetectRatio/* = 2.f*/,
            float init_thres_contrast/* = 0.04f*/, bool up_sample/* = true*/,
            float min_thres_contrast, uint32_t nbAbstractSamples/* = 300u*/)
{
    auto task = [this, src_getter{std::move(src_getter)}, width, height, targetNbKPoints, minOverDetectRatio, init_thres_contrast, up_sample, min_thres_contrast, nbAbstractSamples]() mutable {
        std::lock_guard<std::mutex> lk{_mutex};
        feed(std::move(src_getter), width, height, targetNbKPoints, minOverDetectRatio, init_thres_contrast, up_sample, min_thres_contrast);
        std::tuple<std::vector<KeyPoint>, std::vector<SiftDescriptor>, std::vector<bool>> result = std::make_tuple(
                get_keypoints(), get_descriptors(), _anms_filter.uniformSample(_kpoints.get(), nbAbstractSamples, 1.11f)
        );
        return result;
    };
    return _thread_pool.enqueue(std::move(task));
}