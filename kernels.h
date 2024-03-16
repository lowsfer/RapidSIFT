//
// Created by yao on 12/01/18.
//

#pragma once

#include <cuda_fp16.h>
#include <vector>
#include "types.h"
#include "utils_sift.h"
#include <numeric>
#include <cassert>
#include <device_atomic_functions.h>
#include "public_types.h"

// default width of descriptor histogram array
static const int SIFT_DESCR_WIDTH = sift_desc_width;

// default number of bins per histogram in descriptor array
static const int SIFT_DESCR_HIST_BINS = sift_desc_hist_bins;

// assumed gaussian blur for input image
static const float SIFT_INIT_SIGMA = 0.5f;

// width of border in which to ignore keypoints
static const int SIFT_IMG_BORDER = 5;

// maximum steps of keypoint interpolation before failure
static const int SIFT_MAX_INTERP_STEPS = 5;

// default number of bins in histogram for orientation assignment
static const int SIFT_ORI_HIST_BINS = 36;

// determines gaussian sigma for orientation assignment
static const float SIFT_ORI_SIG_FCTR = 1.5f;

// determines the radius of the region used in orientation assignment
static const float SIFT_ORI_RADIUS = 3 * SIFT_ORI_SIG_FCTR;

// orientation magnitude relative to max that results in new feature
static const float SIFT_ORI_PEAK_RATIO = 0.8f;

// determines the size of a single descriptor orientation histogram
static const float SIFT_DESCR_SCL_FCTR = 3.f;

// threshold on magnitude of elements of descriptor vector
static const float SIFT_DESCR_MAG_THR = 0.2f;

// factor used to convert floating-point descriptor to unsigned char
static const float SIFT_INT_DESCR_FCTR = 512.f;

constexpr int cuda_DoG_max_filter_size = 33;
cudaError_t cuda_DoG(cudaTextureObject_t tex, int2 img_size,
                     const std::vector<float>& filter1d,
                     const pitched_ptr<float>& DoG,
                     const pitched_ptr<float>& blurred,
                     bool downsample,
                     cudaStream_t stream = 0);

//if you need to double the size of input image, simple double img_size will do
cudaError_t cuda_blur(cudaTextureObject_t tex, int2 img_size,
                      const std::vector<float>& filter1d,
                      const pitched_ptr<float>& blurred,
                      cudaStream_t stream = 0);

// implementation based on normal CDF / erfc
std::vector<float> make_gaussian_filter(float sigma, float radiusToSigRatio = 4.f);

struct alignas(8) ScaleSpaceExtrema{
    ushort2 location;
    uint8_t octave;
    uint8_t layer;//idx of DoG layer in the octave
    bool is_max;
};

template <typename T>
struct alignas(128) GPUArray{
    uint32_t count;
    uint32_t padding[31];
    T data[];

#ifdef __CUDACC__
    __device__ __forceinline__ uint32_t push_back(const T& item, const uint32_t& capacity){
        uint32_t idx = atomicAdd(&count, 1);
        if(idx < capacity)
            data[idx] = item;
        return idx;
    }
#endif
};
static_assert(offsetof(GPUArray<float>, data) == 128, "fatal error");

template <typename T>
__global__ void kernel_memset_scalar(T* ptr){
    *ptr = 0u;
};

cudaError_t cuda_find_extrema(
        const cudaTextureObject_t *DoG_layers, int num_DoG_layers,
        int octave_id, int2 img_size,
        GPUArray<ScaleSpaceExtrema> * extremas, uint32_t max_num_extremas, float threshold,
        bool reset_extremas,
        cudaStream_t stream);

cudaError_t cuda_get_num_extremas(
        const GPUArray<ScaleSpaceExtrema>* extremas, uint32_t* num_extremas, cudaStream_t stream);

cudaError_t cuda_get_num_kpoints(
        const GPUArray<KeyPoint>* kpoints, uint32_t* num_kpoints, cudaStream_t stream);

cudaError_t cuda_make_keypoints(const GPUArray<ScaleSpaceExtrema>* extremas,
                                uint32_t num_extremas,
                                const cudaTextureObject_t* DoG_layers, int num_DoG_layers,
                                int octave_id,//for assertions only
                                int2 img_size,
                                int2 loc_lb, int2 loc_ub,
                                float thres_contrast, float thres_edge,
                                float sigma,
                                GPUArray<KeyPoint>* kpoints, uint32_t max_num_kpoints,
                                bool reset_kpoints, cudaStream_t stream);

cudaError_t cuda_assign_orientation(
        const cudaTextureObject_t* gauss_layers,
        int num_octave_layers,//layers per octave.
        int num_octaves,
        int2 octave0_size, // up-scaled size if up-scaling is used
        GPUArray<KeyPoint>* kpoints, uint32_t max_num_kpoints, const uint32_t num_points_before_split,
        bool octave0_is_up_sampled, cudaStream_t stream);

cudaError_t cuda_describe(const cudaTextureObject_t* __restrict__ gauss_layers,
                          int num_octave_layers,//layers per octave.
                          int num_octaves,
                          int2 octave0_size, // up-scaled size if up-scaling is used.
                          bool up_sampled,
                          const GPUArray<KeyPoint>* __restrict__ kpoints,
                          uint32_t num_kpoints,//min(max_num_kpoints, kpoints->count)
                          float max_radius,
                          SiftDescriptor* descriptors,
                          bool useRootSIFT,
                          cudaStream_t stream);
cudaError_t cuda_down_sample(
        const pitched_ptr<float>& dst, uint2 dst_size,
        const pitched_ptr<const float>& src, cudaStream_t stream);



struct alignas(16) Patch32x32{
    static constexpr int32_t patchSize = 32;
    int8_t data[patchSize][patchSize];
};

cudaError_t cuda_makePatch(
    cudaTextureObject_t img, // the orignal image
    int2 imgSize, // the original image size
    const GPUArray<KeyPoint>* __restrict__ kpoints,
    uint32_t num_kpoints,//min(max_num_kpoints, kpoints->count)
    float magFactor,
    float rcpOutQuantScale,
    Patch32x32* const patches,
    cudaStream_t stream);
