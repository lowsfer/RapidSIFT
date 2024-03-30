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
// Created by yao on 8/01/18.
//
#include "SIFT_worker.h"
#include <opencv2/opencv.hpp>
#include "fp16.h"
#include "RapidSift.h"
#include <opencv2/features2d.hpp>
#include "filter/robust_gms_matcher.h"
#include "filter/PropagationMatchFilter.h"
#include <fstream>
#pragma GCC push_options
#pragma GCC optimize ("O0")

DescType descType = DescType::kSOSNet;

const cv::KeyPoint KeyPoint2cvKeyPoint(const KeyPoint& kpt){
    cv::KeyPoint ret;
    ret.pt = cv::Point2f(kpt.location.x, kpt.location.y);
    ret.size = half2float(kpt.size);
    ret.angle = kpt.angle * 360.f / 256.f;
    ret.octave = kpt.octave;
    ret.response = half2float(kpt.response);
    ret.class_id = -1;
    return ret;
}

const cv::Mat desc2cvDesc(const std::vector<SiftDescriptor>& desc){
    static_assert(sizeof(SiftDescriptor) == 128, "fatal error");
    cv::Mat result = cv::Mat::zeros(int(desc.size()), 128, CV_8U);
    result.resize(desc.size());
    for(unsigned i = 0; i < desc.size(); i++){
        for(int j = 0; j < 128; j++)
            result.at<unsigned char>(i, j) = (&desc[i].data[0][0][0])[j];
    }
    return result;
}

bool debug_flag = false;
#if 0
int main(){
//    cv::Mat img = cv::imread("/home/yao/projects/mapper2d/data/DroneMapper_Golf9_May2016/DJI_0006.JPG", cv::IMREAD_GRAYSCALE);
    int2 img_size = {400, 300};
    std::vector<uint8_t> img(img_size.x * img_size.y);

    std::unique_ptr<sift_master> sift = std::make_unique<sift_master>();
    for(int i = 0; i < 2; i++)
    {
        const auto result = sift->detect_and_describe([img, img_size]() { return img.data(); }, img_size.x, img_size.y, true, true).get();

        const auto& kpoints = result.first;
        const auto& descriptors = result.second;
        (void)kpoints;(void)descriptors;
        std::cout << "num_kpoints (after split) = " << kpoints.size() << std::endl;

        std::vector<cv::KeyPoint> cv_kpoints(kpoints.size());
        std::transform(kpoints.begin(), kpoints.end(), cv_kpoints.begin(), KeyPoint2cvKeyPoint);
        cv::Mat img_kpoints = img.clone();
        cv::drawKeypoints(img, cv_kpoints, img_kpoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("image", img);
        cv::imshow("keypoints", img_kpoints);

        checkCudaError(cudaDeviceSynchronize());
        debug_flag = true;
    }
    {
//        std::unique_ptr<SIFT_worker> sift = std::make_unique<SIFT_worker>();
//        require(img.type() == CV_8U);
//        cv::Mat img_rot(img.size(), img.type());
//        for (int i = 0; i < img.rows; i++) {
//            for (int j = 0; j < img.cols; j++) {
//                img_rot.at<uint8_t>(img.rows - 1 - i, img.cols - 1 - j) = img.at<uint8_t>(i, j);
//            }
//        }
//        sift->feed([img_rot]() { return img_rot.data; }, img_rot.cols, img_rot.rows, true);
//        const auto kpoints_rot = sift->get_keypoints(false).get();
//        const auto descriptors_rot = sift->get_descriptors().get();
//        std::cout << "num_kpoints (after split) = " << kpoints_rot.size() << std::endl;
//
//        std::vector<cv::KeyPoint> cv_kpoints_rot(kpoints_rot.size());
//        std::transform(kpoints_rot.begin(), kpoints_rot.end(), cv_kpoints_rot.begin(), KeyPoint2cvKeyPoint);
//        cv::Mat img_kpoints_rot = img_rot.clone();
//        cv::drawKeypoints(img_rot, cv_kpoints_rot, img_kpoints_rot, cv::Scalar::all(-1),
//                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
//        cv::imshow("image_rot", img_rot);
//        cv::imshow("keypoints_rot", img_kpoints_rot);
        checkCudaError(cudaDeviceSynchronize());
    }
//    while('q' != cv::waitKey(0))
//        ;


//    cudaDeviceReset();
    return 0;
}
#endif

std::vector<cv::DMatch> correlationMatch(const cv::Mat& srcA, const cv::Mat& srcB, int lb, int ub)
{
#define COL_SHIFT 0
#define ROW_SHIFT 1
#define ROW_NORMALIZE 0

    cv::Mat a;
    srcA.convertTo(a, CV_32F);
#if COL_SHIFT
    for (int i = 0; i < a.cols; i++) {
        // looks like this is better than row-wise mean, when combined with normalize
        a.col(i) -= cv::mean(a.col(i));
    }
#endif
#if ROW_SHIFT
    for (int i = 0; i < a.rows; i++) {
        a.row(i) -= cv::mean(a.row(i));
    }
#endif
    a.convertTo(a, CV_8S);
    a = cv::max(cv::min(a, ub), lb);
    a.convertTo(a, CV_32F);
#if ROW_NORMALIZE
    for (int i = 0; i < a.rows; i++) {
        cv::normalize(a.row(i), a.row(i));
    }
//    a = cv::max(cv::min(a, 0.2), -0.2);
#endif
    cv::Mat b;
    srcB.convertTo(b, CV_32F);
#if COL_SHIFT
    for (int i = 0; i < b.cols; i++) {
        b.col(i) -= cv::mean(b.col(i));
    }
#endif
#if ROW_SHIFT
    for (int i = 0; i < b.rows; i++) {
        b.row(i) -= cv::mean(b.row(i));
    }
#endif
    b.convertTo(b, CV_8S);
    b = cv::max(cv::min(b, ub), lb);
    b.convertTo(b, CV_32F);
#if ROW_NORMALIZE
    for (int i = 0; i < b.rows; i++) {
        cv::normalize(b.row(i), b.row(i));
    }
//    b = cv::max(cv::min(b, 0.2), -0.2);
#endif
    cv::Mat c = a * b.t();
    std::vector<cv::DMatch> result(a.rows);
    for (int i = 0; i < a.rows; i++) {
        cv::DMatch& m = result.at(i);
        m = cv::DMatch(i, 0, -INFINITY);
        for (int j = 1; j < b.rows; j++) {
            if (c.at<float>(i, j) > m.distance) {
                m.trainIdx = j;
                m.distance = c.at<float>(i, j);
            }
        }
    }
    return result;
}
// this gives slightly better descriptors.
void describeWithOpenCV(std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>& kpts, cv::Mat const& img, bool upsample)
{
    UNUSED(kpts); UNUSED(img); UNUSED(upsample);
    auto cv_sift = cv::SIFT::create();
    std::vector<cv::KeyPoint> cv_kpoints(kpts.first.size());
    std::transform(kpts.first.begin(), kpts.first.end(), cv_kpoints.begin(), [upsample](KeyPoint const& p){
        auto p_ = KeyPoint2cvKeyPoint(p);
        p_.pt.x -= 0.25f;
        p_.pt.y -= 0.25f;
        int octave = p.octave; // (upsample ? p.octave + 1 : p.octave);
        // float delta = (p.layer_offset + 0.5f) / 256.f;
        p_.octave = octave + ((int)(octave >= 0 ? p.layer : p.layer + 1) << 8);
        return p_;
    });
    cv::Mat cv_descriptors;
    cv_sift->detectAndCompute(img, cv::noArray(), cv_kpoints, cv_descriptors, true);
    assert(cv_descriptors.rows == (int)cv_kpoints.size());
    assert(cv_descriptors.type() == CV_32F);
    // cv_sift->compute(img, cv_kpoints, cv_descriptors);
    for (size_t i = 0; i < kpts.first.size(); i++) {
        for(int j = 0; j < 128; j++)
            (&kpts.second.at(i).data[0][0][0])[j] = (uint8_t)cv_descriptors.at<float>(i, j);
    }
}

void removeUpsample(std::vector<cv::KeyPoint>& kpts, cv::Mat& desc)
{
    std::vector<cv::KeyPoint> p;
    cv::Mat d = desc.clone();
    for (size_t i = 0; i < kpts.size(); i++)
    {
        auto const& k = kpts[i];
        if ((k.octave & 255) < 128) {
            p.push_back(k);
            desc.row(i).copyTo(d.row(p.size() - 1));
        }
    }
    kpts = std::move(p);
    desc = d.rowRange(0, kpts.size());
}


#if 1
int main(){
//    checkCudaError(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)); //this saves CPU
    checkCudaError(cudaSetDeviceFlags(cudaDeviceScheduleSpin));

    // cv::Mat img = cv::imread("/home/yao/projects/3D/data/users/a/images/43.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat img2 = cv::imread("/home/yao/projects/3D/data/users/a/images/237.jpg", cv::IMREAD_GRAYSCALE);

//    cv::Mat img = cv::imread("/home/yao/projects/rapidsift/images/data/img1.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/rapidsift/images/oblique/IMG_7735.JPG", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/CudaSift-Pascal/data/img1.png", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/mapper2d/data/DroneMapper_Golf9_May2016/DJI_0601.JPG", cv::IMREAD_GRAYSCALE);
    // cv::Mat img = cv::imread("/home/yao/projects/mapper2d/data/golf_small/DJI_0601.JPG", cv::IMREAD_GRAYSCALE);
    // cv::Mat img = cv::imread("/home/yao/projects/3D/data/village/undistorted_small/IMG_7721.JPG", cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread("/home/yao/projects/3D/data/users/c/images/304.jpg", cv::IMREAD_GRAYSCALE); // need undistortion
//    cv::Mat img = cv::imread("/home/yao/projects/mapper2d/data/golf_small/DJI_0012.JPG", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/rapidsfm2/build/cache/a.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/rapidsift/images/dot.bmp", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/mapper2d/data/DroneMapper_Golf9_May2016/DJI_0006.JPG", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/rapidsift/images/sunflower1.jpg", cv::IMREAD_GRAYSCALE);
//    cv::Mat img = cv::imread("/home/yao/projects/rapidsift/images/EtaCarinae.jpg", cv::IMREAD_GRAYSCALE);
//    cv::resize(img, img, cv::Size{1280, 960});
//    cv::resize(img, img, cv::Size{400, 300});
    //cv::resize(img, img, cv::Size{960, 540});
    if(img.empty()){
        throw std::runtime_error("image load failure");
    }
    const float contrast_threshold = 0.04f;
    float const minContrastThres = 0.002f;
    static_cast<void>(minContrastThres);
    const bool up_sample = false;

    const float2 refPt = {611, 509};

    const uint32_t targetNbKPoints = 6000; (void)targetNbKPoints;
    const float minOverDetectRatio = 4.f; (void)minOverDetectRatio;
    std::unique_ptr<RapidSift> sift{create_sift(1, descType)};
    if(true)
    {
        auto result = sift->detect_and_describe([img]() { return img.data; }, img.cols, img.rows, contrast_threshold, up_sample).get();
        // auto result = sift->uniform_detect_and_compute([img]() { return img.data; }, img.cols, img.rows, targetNbKPoints, minOverDetectRatio, contrast_threshold, up_sample, minContrastThres).get();
        // describeWithOpenCV(result, img, up_sample);
        result = sort_kpoints(result);

        const auto& kpoints = result.first;
        const auto& descriptors = result.second;
        (void)descriptors;

        std::vector<cv::KeyPoint> cv_kpoints(kpoints.size());
        std::transform(kpoints.begin(), kpoints.end(), cv_kpoints.begin(), KeyPoint2cvKeyPoint);
        for (unsigned i = 0; i < kpoints.size(); i++){
            const auto& kpt = kpoints.at(i);
            if (kpt.location.y > img.rows){
                printf("%u %lu : %f, %f\n", i, kpoints.size(), kpt.location.x, kpt.location.y);
            }
            require(in_range(kpt.location.x, 0.f, (float)img.cols));
            require(in_range(kpt.location.y, 0.f, (float)img.rows));
        }

        {
            cv::Mat img_kpoints = img.clone();
            cv::drawKeypoints(img, cv_kpoints, img_kpoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            for (auto const &k : cv_kpoints)
            {
                if (sqr(k.pt.x - refPt.x) + sqr(k.pt.y - refPt.y) < sqr(5))
                {
                    printf("####[rapidsift]: (%f, %f) size = %f, response = %f, octave = %d *********\n", k.pt.x, k.pt.y, k.size, k.response, k.octave);
                }
            }
            cv::imshow("keypoints_rapidsift", img_kpoints);
            std::cout << "num_kpoints for rapidsift: " << cv_kpoints.size() << std::endl;
        }
//        return 0;
        auto cv_descriptors = desc2cvDesc(descriptors);
#define USE_CV_SIFT 0
#if USE_CV_SIFT
        auto cv_sift = cv::SIFT::create(0, 3, contrast_threshold);
        cv_kpoints.clear();
        cv_descriptors.release();
        cv_sift->detectAndCompute(img, cv::noArray(), cv_kpoints, cv_descriptors);
        if (!up_sample)
        {
            removeUpsample(cv_kpoints, cv_descriptors);
        }
        {
            cv::Mat img_kpoints = img.clone();
            cv::drawKeypoints(img, cv_kpoints, img_kpoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            for (auto const &k : cv_kpoints)
            {
                if (sqr(k.pt.x - refPt.x) + sqr(k.pt.y - refPt.y) < sqr(5))
                {
                    printf("####[opencv]: (%f, %f) size = %f, response = %f *********\n", k.pt.x, k.pt.y, k.size, k.response);
                }
            }
            cv::imshow("keypoints_opencv", img_kpoints);
            std::cout << "num_kpoints for opencv: " << cv_kpoints.size() << std::endl;
        }
#endif

        std::cout << "num_kpoints (after split) = " << cv_kpoints.size() << std::endl;
//        return 0;

        std::unique_ptr<RapidSift>& sift2 = sift;
        require(img.type() == CV_8U);
#if 0
        cv::Mat img2(img.size(), img.type());
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                img2.at<uint8_t>(img.rows - 1 - i, img.cols - 1 - j) = img.at<uint8_t>(i, j);
            }
        }
#else
//        cv::Mat img2 = cv::imread("/home/yao/projects/rapidsift/images/data/img2.jpg", cv::IMREAD_GRAYSCALE);
//        cv::Mat img2 = cv::imread("/home/yao/projects/rapidsift/images/oblique/IMG_7736.JPG", cv::IMREAD_GRAYSCALE);
//        cv::Mat img2 = cv::imread("/home/yao/projects/CudaSift-Pascal/data/img2.png", cv::IMREAD_GRAYSCALE);
//        cv::Mat img2 = cv::imread("/home/yao/projects/mapper2d/data/DroneMapper_Golf9_May2016/DJI_0618.JPG", cv::IMREAD_GRAYSCALE);
        // cv::Mat img2 = cv::imread("/home/yao/projects/mapper2d/data/golf_small/DJI_0618.JPG", cv::IMREAD_GRAYSCALE);
        // cv::Mat img2 = cv::imread("/home/yao/projects/3D/data/village/undistorted_small/IMG_7722.JPG", cv::IMREAD_GRAYSCALE);
        cv::Mat img2 = cv::imread("/home/yao/projects/3D/data/users/c/images/301.jpg", cv::IMREAD_GRAYSCALE);
//        cv::Mat img2 = cv::imread("/home/yao/projects/mapper2d/data/golf_small/DJI_0017.JPG", cv::IMREAD_GRAYSCALE);
//        cv::Mat img2 = cv::imread("/home/yao/projects/rapidsift/images/dotr.bmp", cv::IMREAD_GRAYSCALE);
//        cv::Mat img2 = cv::imread("/home/yao/projects/rapidsfm2/build/cache/b.jpg", cv::IMREAD_GRAYSCALE);
    //    cv::flip(img, img2, -1);
#endif
        auto result2 = sift2->detect_and_describe([img2]() { return img2.data; }, img2.cols, img2.rows, contrast_threshold, up_sample).get();
        // auto result2 = sift2->uniform_detect_and_compute([img2]() { return img2.data; }, img2.cols, img2.rows, targetNbKPoints, minOverDetectRatio, contrast_threshold, up_sample, minContrastThres).get();
        // describeWithOpenCV(result2, img2, up_sample);
        result2 = sort_kpoints(result2);
        const auto& kpoints2 = result2.first;
        const auto& descriptors2 = result2.second;
        (void)descriptors2;
//        return 0;

        std::vector<cv::KeyPoint> cv_kpoints2(kpoints2.size());
        std::transform(kpoints2.begin(), kpoints2.end(), cv_kpoints2.begin(), KeyPoint2cvKeyPoint);
        auto cv_descriptors2 = desc2cvDesc(descriptors2);
        checkCudaError(cudaDeviceSynchronize());

#if USE_CV_SIFT
        cv_kpoints2.clear();
        cv_descriptors2.release();
        cv_sift->detectAndCompute(img2, cv::noArray(), cv_kpoints2, cv_descriptors2);
        if (!up_sample)
        {
            removeUpsample(cv_kpoints2, cv_descriptors2);
        }
#endif
        std::cout << "num_kpoints (after split) = " << cv_kpoints2.size() << std::endl;

#if 0
        {
            cv_kpoints.clear();
            cv_kpoints2.clear();
            cv_descriptors.deallocate();
            cv_descriptors2.deallocate();
            auto cvSIFT = cv::SIFT::create(0, 3, 0.04);
            cvSIFT->detectAndCompute(img, cv::Mat(), cv_kpoints, cv_descriptors);
            if (!up_sample)
            {
                removeUpsample(cv_kpoints, cv_descriptors);
            }
            cvSIFT->detectAndCompute(img2, cv::Mat(), cv_kpoints2, cv_descriptors2);
            if (!up_sample)
            {
                removeUpsample(cv_kpoints2, cv_descriptors2);
            }
            std::cout << "opencv sift points: " << cv_kpoints.size() << " " << cv_kpoints2.size() << std::endl;
        }
#endif
        cv::Mat img_kpoints = img.clone();
        cv::drawKeypoints(img, cv_kpoints, img_kpoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("image", img);
        cv::imshow("keypoints", img_kpoints);

        cv::Mat img_kpoints2 = img2.clone();
        cv::drawKeypoints(img2, cv_kpoints2, img_kpoints2, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("image2", img2);
        cv::imshow("keypoints2", img_kpoints2);

#define USE_CORRELATION_MATCH 0
#define USE_DIV 0
        // try less bits for sift coefficients
#if USE_DIV
        // 64 or even 128 looks not too bad. Seems like one or two bits are enough for every coefficient of SIFT descriptor!
        // 16 works very well with only ~0.4% less correct matches than int8.
        // This means we can use int4. Use TensorCore!
        const int divFactor = 16;
        cv::Mat(cv_descriptors/divFactor - 0.5f).convertTo(cv_descriptors, CV_8U);
        cv::Mat(cv_descriptors2/divFactor - 0.5f).convertTo(cv_descriptors2, CV_8U);
#endif
        std::vector<cv::DMatch> matches;
#if !USE_CORRELATION_MATCH
        const bool crossCheck = true;
        auto matcher = cv::BFMatcher::create(cv::NORM_L2, crossCheck);
        matcher->match(cv_descriptors, cv_descriptors2, matches, cv::noArray());
#else
        matches = correlationMatch(cv_descriptors, cv_descriptors2, -8, 7);///////////@fixme: algorithm test code
#endif
        std::vector<unsigned char> mask(matches.size(), (unsigned char)255u);
        if (matches.size() > 4)
        {
            std::vector<cv::Point2f> src, dst;
            for (const auto& m : matches){
                src.push_back(cv_kpoints[m.queryIdx].pt);
                dst.push_back(cv_kpoints2[m.trainIdx].pt);
            }
            if (true)
            {
                cv::findFundamentalMat(src, dst, cv::USAC_DEFAULT, 6.f, 0.999, mask);
            }
            else
            {
                vec2<int> imgSize{img.cols, img.rows};
                match_filters::PropagationMatchFilter filter{&cv_kpoints, imgSize, &cv_kpoints2, &matches};
                std::vector<bool> boolMask = filter.getInlierMask();
                for (size_t i = 0; i < matches.size(); i++)
                    mask.at(i) = boolMask.at(i) ? 255 : 0;
                cv::findFundamentalMat(src, dst, cv::USAC_DEFAULT, 6.f, 0.995, mask);
            }
            const auto correct = matches.size() - std::count(mask.begin(), mask.end(), 0);
            std::cout << "correct matched points: " << correct << " (" << float(correct)/std::min(cv_kpoints.size(), cv_kpoints2.size()) * 100 << "%) from " << matches.size() << " matches" << std::endl;

            std::vector<uchar> maskNoMatch(cv_kpoints.size(), uchar{1});
            std::vector<uchar> maskNoMatch2(cv_kpoints2.size(), uchar{1});
            std::ofstream fout("./matches.txt", std::ios::trunc);
            for (unsigned i = 0; i < matches.size(); i++){
                if (mask[i]){
                    const auto m = matches[i];
                    maskNoMatch.at(m.queryIdx) = 0;
                    maskNoMatch2.at(m.trainIdx) = 0;
                    fout << int(int8_t(uint8_t(cv_kpoints[m.queryIdx].octave & 0xFF))) <<  ", " << (cv_kpoints[m.queryIdx].pt.x + cv_kpoints2[m.trainIdx].pt.x - img2.cols) / 2 << ", " << (cv_kpoints[m.queryIdx].pt.y + cv_kpoints2[m.trainIdx].pt.y - img2.rows) / 2 << ", " << fmod(cv_kpoints[m.queryIdx].angle + 180.f - cv_kpoints2[m.trainIdx].angle + 360.f*2, 360.f) << std::endl;
                }
            }
#if 0
            std::vector<cv::KeyPoint> kptsNoMatch, kptsNoMatch2;
            std::vector<SiftDescriptor> descNoMatch, descNoMatch2;
            for (size_t i = 0; i < cv_kpoints.size(); i++) {
                if (maskNoMatch.at(i)) {
                    kptsNoMatch.push_back(cv_kpoints.at(i));
                    descNoMatch.push_back(descriptors.at(i));
                }
            }
            for (size_t i = 0; i < cv_kpoints2.size(); i++) {
                if (maskNoMatch2.at(i)) {
                    kptsNoMatch2.push_back(cv_kpoints2.at(i));
                    descNoMatch2.push_back(descriptors2.at(i));
                }
            }
            cv::Mat imgNoMatch, imgNoMatch2;
            cv::drawKeypoints(img, kptsNoMatch, imgNoMatch, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::drawKeypoints(img2, kptsNoMatch2, imgNoMatch2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imshow("No Match (left)", imgNoMatch);
            cv::imshow("No Match (right)", imgNoMatch2);
#endif
        }

        matches.erase(std::remove_if(matches.begin(), matches.end(), [&](const cv::DMatch& a)->bool{return !mask[&a - matches.data()];}), matches.end());
        cv::Mat img_matches;
        std::vector<char> drawMask(matches.size(), char{1});
#if 0
        for (size_t i = 0; i < matches.size(); i++) {
            auto const& p = cv_kpoints.at(matches.at(i).queryIdx).pt;
            if (p.y > 1.02f * (p.x-705))
            {
                drawMask.at(i) = false;
            }
        }
        printf("Drawing %u matches\n", (uint32_t)std::count(drawMask.begin(), drawMask.end(), char{1}));
#endif
        cv::drawMatches(img, cv_kpoints, img2, cv_kpoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), drawMask, cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("matches", img_matches);

        while('q' != cv::waitKey(0))
            ;
    }
    //benchmark
    if (true)
    {
        std::unique_ptr<CUevent_st, cuda_event_deleter> event_start{make_cuda_event(cudaEventDefault)};
        std::unique_ptr<CUevent_st, cuda_event_deleter> event_end{make_cuda_event(cudaEventDefault)};

        std::unique_ptr<RapidSift> sift{create_sift(4, descType)};
        cv::Mat img = cv::imread("/home/yao/projects/mapper2d/data/DroneMapper_Golf9_May2016/DJI_0601.JPG", cv::IMREAD_GRAYSCALE);

        //warm-up
        if(false) {
            for (int i = 0; i < 4; i++) {
                sift->detect_and_describe([img]() { return img.data; }, img.cols, img.rows, up_sample).get();
            }
        }

        const int repeat = 128;
        std::queue<std::future<std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>>> results;
        checkCudaError(cudaEventRecord(event_start.get(), 0));
        for (int i = 0; i < repeat; i++) {
            results.emplace(
                    sift->detect_and_describe([img]() { return img.data; }, img.cols, img.rows, contrast_threshold, up_sample));
            if(results.size() > sift->get_num_workers() * 4) {
                auto result = results.front().get();
//                printf("num_kpoints = %lu\n", result.first.size());
                results.pop();
            }
        }
        while(!results.empty()) {
            auto result = results.front().get();
            printf("num_kpoints = %lu\n", result.first.size());
            results.pop();
        }
        checkCudaError(cudaEventRecord(event_end.get(), 0));
        checkCudaError(cudaDeviceSynchronize());
        float duration = 0;
        checkCudaError(cudaEventElapsedTime(&duration, event_start.get(), event_end.get()));
        printf("FPS = %f\n", repeat * 1000 / duration);
    }

//    cudaDeviceReset();
    return 0;
}
#endif
#pragma GCC pop_options
