#include "kernels.h"

std::vector<float> make_gaussian_filter(float sigma, float radiusToSigRatio){
#if 1
    // Sampled Gaussian kernel
    // This is incorrect but the result is better. i.e. more key points. OpenCV GaussianBlur is also doing this.
    (void)(radiusToSigRatio);
    assert(sigma > 0);
    const int filter_size = std::min(int(std::round(sigma * radiusToSigRatio * 2 + 1)) | 1, cuda_DoG_max_filter_size);
    const int radius = (filter_size - 1) / 2;
    std::vector<float> filter((size_t)filter_size);
    for(int i = 0; i < filter_size; i++){
        filter[i] = std::exp(-sqr(i - radius) / (2 * sqr(sigma)));
    }
    float const factor = 1.f / std::accumulate(filter.begin(), filter.end(), 0.f);
    // printf("filter sigma=%f: ", sigma);
    for(int i = 0; i < filter_size; i++){
        filter[i] *= factor;
        // printf("%f, ", filter[i]);
    }
    // printf("\n");
    return filter;
#else
    // Ideal implementation based on normal CDF / erfc. This is theoretically better but detects less key points.
    assert(sigma > 0);
    const int filter_size = std::min(int(std::round(sigma * radiusToSigRatio * 2 + 1)) | 1, cuda_DoG_max_filter_size);
    const int radius = (filter_size - 1) / 2;

    std::vector<float> filter((size_t)filter_size);

    double const rcpSigma = 1.0 / sigma;
    auto const normalCDFx2 = [rcpSigma](double x){
        double const rsqrt_2 = std::sqrt(0.5);
        double const factor = -rsqrt_2 * rcpSigma;
        return erfc(x * factor);
    };

    double left = normalCDFx2(-0.5 - radius);
    for(int i = 0; i < filter_size; i++){
        double const right = normalCDFx2(0.5 - radius + i);
        filter[i] = right - left;
        left = right;
    }
    const float factor = static_cast<float>(1.0 / std::accumulate(filter.begin(), filter.end(), 0.0));
    // printf("filter sigma=%f: ", sigma);
    for(int i = 0; i < filter_size; i++){
        filter[i] *= factor;
        // printf("%f, ", filter[i]);
    }
    // printf("\n");
    return filter;
#endif
}
