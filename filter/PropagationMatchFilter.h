//
// Created by yao on 17/06/18.
//

#pragma once
#include <opencv2/opencv.hpp>
#include "types.h"
#include "geometry.h"
#include "ransac.hpp"

namespace match_filters {
class PropagationMatchFilter {
public:
    static constexpr int halo = 1;
    static constexpr float threshold_relative = 0.15f;
    static constexpr float ransac_confidence = 0.25f;
    static constexpr size_t ransac_max_iterations = 32;
public:
    PropagationMatchFilter() = default;
    PropagationMatchFilter(const std::vector<cv::KeyPoint> *kpts0, vec2<int> imgSize0,
                       const std::vector<cv::KeyPoint> *kpts1, const std::vector<cv::DMatch> *matches) {
        init(kpts0, imgSize0, kpts1, matches);
    }

    void init(const std::vector<cv::KeyPoint> *kpts0, vec2<int> imgSize0, const std::vector<cv::KeyPoint> *kpts1,
              const std::vector<cv::DMatch> *matches);

    std::vector<bool> getInlierMask(int minVotes = 2) const;

private:
    struct Params {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        affine2_t<float> trans;
        uint32_t numInliers;
    };

    const std::vector<uint32_t> &cell(int i, int j) const { return mCells.at(i * mCols + j); }

    std::vector<uint32_t> &cell(int i, int j) { return mCells.at(i * mCols + j); }

    const Params &param(int i, int j) const { return mParams.at(i * mCols + j); }

    Params &param(int i, int j) { return mParams.at(i * mCols + j); }

    enum Direction {
        Forward,
        Backward
    };

    void processCell(uint32_t y, uint32_t x, Direction direction, bool vote = false);

    void propagate(Direction direction, bool vote = false);

    void run();

private:
    uint32_t mCellWidth = 0;
    std::vector<std::vector<uint32_t>> mCells;
    uint32_t mCols = 0;
    uint32_t mRows = 0;
    const std::vector<cv::KeyPoint> *mKpts0 = nullptr;
    const std::vector<cv::KeyPoint> *mKpts1 = nullptr;
    const std::vector<cv::DMatch> *mMatches = nullptr;
    std::vector<Params, Eigen::aligned_allocator<Params>> mParams;
    std::vector<int> mVotes;
private: //buffers used to reduce new/delete
    std::vector<uint32_t> input;
};
}