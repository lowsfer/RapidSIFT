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
// Created by yao on 11/01/18.
//

#include <boost/test/unit_test.hpp>
#include <random>
#include "../conv1d.h"
#include "../utils_sift.h"
#include "../types.h"
#include <opencv2/opencv.hpp>
#include "../kernels.h"
#include "../utils_host.h"
#include <cuda_runtime.h>

BOOST_AUTO_TEST_CASE(conv1d_test)
{
    constexpr int radius = 2;
    conv1d<radius> conv;
    const float filter[radius * 2 + 1] = {.1f, .2f, .3f, .4f, .35f};
    conv.set_filter(filter);

    std::default_random_engine engine;
    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    const int input_size = 10;
    std::unique_ptr<float[]> input_data{new float[input_size]};//use heap memory to help valgrind
    float (&input)[input_size] = *reinterpret_cast<float (*)[input_size]>(input_data.get());
    for (int i = 0; i < input_size; i++)
        input[i] = dist(engine);

    const int output_size = input_size + 1 - conv.filter_size;

    std::unique_ptr<float[]> output_ref_data{new float[output_size]};
    float (&output_ref)[output_size] = *reinterpret_cast<float (*)[output_size]>(output_ref_data.get());
    for (int i = 0; i < output_size; i++) {
        float &result = output_ref[i];
        result = 0.f;
        for (int j = 0; j < conv.filter_size; j++) {
            result += input[i + j] * conv.filter(j);
        }
    }

    std::unique_ptr<float[]> output_data{new float[output_size]};
    float (&output)[output_size] = *reinterpret_cast<float (*)[output_size]>(output_data.get());

    conv.init(*reinterpret_cast<const float (*)[conv.filter_size]>(&input[0]));
    int n;
    for (n = 1; n < input_size / conv.filter_size; n++)
        conv(*reinterpret_cast<const float (*)[conv.filter_size]>(&input[conv.filter_size * n]),
             *reinterpret_cast<float (*)[conv.filter_size]>(&output[conv.filter_size * (n - 1)]));
    conv(*reinterpret_cast<const float (*)[conv.filter_size]>(&input[conv.filter_size * n]),
         *reinterpret_cast<float (*)[conv.filter_size]>(&output[conv.filter_size * (n - 1)]),
         input_size % conv.filter_size);



    for (int i = 0; i < input_size + 1 - conv.filter_size; i++)
        BOOST_CHECK_CLOSE(output_ref[i], output[i], 1E-5f);
}

BOOST_AUTO_TEST_CASE(DoG_test)
{
    for(int filter_radius = 1; filter_radius <= cuda_DoG_max_filter_size / 2; filter_radius++) {
        const int filter_size = filter_radius * 2 + 1;
        cv::Mat host_img = cv::imread("/home/yao/projects/mapper2d/data/DroneMapper_Golf9_May2016/DJI_0001.JPG",
                                      cv::IMREAD_GRAYSCALE);
        BOOST_REQUIRE(!host_img.empty());
//        cv::reserve_impl(host_img, host_img, cv::Size{1024, 768});
        host_img.convertTo(host_img, CV_32F, 1.f / 255, 0);
//        cv::resize(host_img, host_img, cv::Size{4000, 12000});

        const int width = host_img.cols;
        const int height = host_img.rows;

//        for(int i = 0; i < width * height; i++){
//            host_img.at<float>(i) = 0.5f;
//        }

        cv::Mat blurred_ref;
        cv::GaussianBlur(host_img, blurred_ref, cv::Size{filter_size, filter_size}, 1.6, 1.6, cv::BORDER_REFLECT);
        cv::Mat cv_filter = cv::getGaussianKernel(filter_size, 1.6, CV_32F);

        float *gpu_img;
        size_t pitch_img;
        checkCudaError(
                cudaMallocPitch((void **) &gpu_img, &pitch_img, sizeof(float) * (size_t) width, (size_t) height));
        const pitched_ptr<float> img{gpu_img, (int) pitch_img};

        checkCudaError(
                cudaMemcpy2D(img.ptr, (size_t) img.pitch, host_img.data, sizeof(float) * width, sizeof(float) * width,
                             (size_t) height, cudaMemcpyHostToDevice));

        float *gpu_DoG;
        size_t pitch_DoG;
        checkCudaError(
                cudaMallocPitch((void **) &gpu_DoG, &pitch_DoG, sizeof(float) * (size_t) width, (size_t) height));
        const pitched_ptr<float> DoG{gpu_DoG, (int) pitch_DoG};

        float *gpu_blurred;
        size_t pitch_blurred;
        const int width_blurred = width;
        const int height_blurred = height;
        checkCudaError(cudaMallocPitch((void **) &gpu_blurred, &pitch_blurred, sizeof(float) * (size_t) width_blurred,
                                       (size_t) height_blurred));
        const pitched_ptr<float> blurred{gpu_blurred, (int) pitch_blurred};

        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceType::cudaResourceTypePitch2D;
        res_desc.res.pitch2D = {
                reinterpret_cast<void *>(gpu_img),
                cudaCreateChannelDesc<float>(),
                (size_t) width,
                (size_t) height,
                pitch_img
        };
        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        //mirror and warp mode are supported only with normalzied coordinate
        for (auto &e: tex_desc.addressMode) e = cudaTextureAddressMode::cudaAddressModeMirror;
        tex_desc.filterMode = cudaTextureFilterMode::cudaFilterModePoint;
        tex_desc.readMode = cudaTextureReadMode::cudaReadModeElementType;
        tex_desc.normalizedCoords = 1;
        cudaTextureObject_t tex;
        checkCudaError(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));

//        const std::vector<float> filter_half = {0.0144f, 0.0447f, 0.1152f, 0.2031f, 0.2453f};
        std::vector<float> filter1d(&cv_filter.at<float>(0), &cv_filter.at<float>(filter_size));

        cudaStream_t stream = nullptr;
        cudaEvent_t event_start, event_end;
        checkCudaError(cudaEventCreate(&event_start));
        checkCudaError(cudaEventCreate(&event_end));
//        for(int i = 0; i < 1000; i++) {//warm up
//            checkCudaError(cuda_DoG(tex, int2{width, height}, filter1d, filter_size, DoG, blurred, false, stream));
//        }
        checkCudaError(cudaEventRecord(event_start, stream));
        const int repeats = 10;
        for (int i = 0; i < repeats; i++) {
            checkCudaError(cuda_DoG(tex, int2{width, height}, filter1d, DoG, blurred, false, stream));
        }
        checkCudaError(cudaEventRecord(event_end, stream));
        float time;
        checkCudaError(cudaStreamSynchronize(stream));
        checkCudaError(cudaEventElapsedTime(&time, event_start, event_end));

        std::cout << "Performance: " << (width * height) * repeats / (time * 1000.f) << " MPixels/sec for filter_size = " << filter_size << std::endl;

        cv::Mat host_blurred(height_blurred, width_blurred, CV_32F);
        cv::Mat host_DoG(height, width, CV_32F);
        checkCudaError(cudaMemcpy2D(host_blurred.data, sizeof(float) * width_blurred,
                                    blurred.ptr, blurred.pitch, sizeof(float) * width_blurred, height_blurred,
                                    cudaMemcpyDeviceToHost));
        checkCudaError(cudaMemcpy2D(host_DoG.data, sizeof(float) * width,
                                    DoG.ptr, DoG.pitch, sizeof(float) * width, height, cudaMemcpyDeviceToHost));

        cudaFree(gpu_img);
        cudaFree(gpu_blurred);
        cudaFree(gpu_DoG);

        double error_max, error_min;

        cv::Mat diff_blur = host_blurred - blurred_ref;
        cv::minMaxLoc(diff_blur, &error_min, &error_max);
        BOOST_CHECK_SMALL((float) error_max, 1E-4f);
        BOOST_CHECK_SMALL((float) error_min, 1E-4f);
//    printf("Error range of guassian blur: [%lf, %lf]\n", error_min, error_max);

        cv::Mat error_DoG = host_DoG - (host_img - host_blurred);
        cv::minMaxLoc(error_DoG, &error_min, &error_max);
        BOOST_CHECK_SMALL((float) error_max, 1E-4f);
        BOOST_CHECK_SMALL((float) error_min, 1E-4f);
//    printf("Error range of DoG: [%lf, %lf]\n", error_min, error_max);

//        cv::imshow("img", host_img);
//        cv::imshow("blurred", host_blurred);
//        cv::imshow("blurred_ref", blurred_ref);
//        cv::imshow("DoG", cv::abs(host_DoG));
//        cv::imshow("blurred_diff", cv::abs(host_blurred - blurred_ref) * 100.f);
//        while(cv::waitKey(0) != 'q')
//            ;
    }
}

BOOST_AUTO_TEST_CASE(make_gaussian_filter_test)
{
    const auto filter = make_gaussian_filter(1.6f);
//    for(auto item : filter)
//        std::cout << item << "\t";
//    std::cout << std::endl;
    BOOST_CHECK_CLOSE(std::accumulate(filter.begin(), filter.end(), 0.f), 1.f, 1E-4f);
}