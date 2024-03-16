//
// Created by yao on 17/06/18.
//

#include "PropagationMatchFilter.h"

namespace match_filters
{

void PropagationMatchFilter::init(const std::vector<cv::KeyPoint>* kpts0, vec2<int> imgSize0, const std::vector<cv::KeyPoint>* kpts1, const std::vector<cv::DMatch>* matches)
{
    if(imgSize0.x >= imgSize0.y) {
        mCols = 16;
        mCellWidth = unsigned(imgSize0.x + mCols - 1) / mCols;
        mRows = unsigned(imgSize0.y + mCellWidth - 1) / mCellWidth;
    }else{
        mRows = 16;
        mCellWidth = unsigned(imgSize0.y + mRows - 1) / mRows;
        mCols = unsigned(imgSize0.x + mCellWidth - 1) / mCellWidth;
    }
    mCells.resize(mCols * mRows);
    for(auto& cell : mCells)
        cell.clear();
    mKpts0 = kpts0;
    mKpts1 = kpts1;
    mMatches = matches;
    const float cellWidthInv = 1.f / mCellWidth;
    for(unsigned i = 0; i < matches->size(); i++)
    {
        const auto& m = (*matches)[i];
        const auto& pt = (*kpts0)[m.queryIdx].pt;
        cell(int(pt.y * cellWidthInv), int(pt.x * cellWidthInv)).push_back(i);
    }
    mParams.resize(mCols * mRows);
    for(auto& param : mParams)
        param = {affine2_t<float>::Identity(), 0};
    mVotes.resize(mMatches->size());
    std::fill(mVotes.begin(), mVotes.end(), 0);

    run();
}

void PropagationMatchFilter::processCell(uint32_t y, uint32_t x, Direction direction, bool vote)
{
//    std::cout << "Processing cell (" << y << ", " << x << ")" << std::endl;
    input.clear();
    for (int i = int(y) - halo; i <= int(y) + halo; i++)
        for (int j = int(x) - halo; j <= int(x) + halo; j++)
            if(i >= 0 && i < int(mRows) && j >= 0 && j < int(mCols))
                input.insert(input.end(), cell(i, j).begin(), cell(i, j).end());
    if(input.size() < 8) {
        param(y,x) = {affine2_t<float>::Identity(), 0};
        return;
    }
    Eigen::ArrayXf point_pairs[4];
    Eigen::ArrayXf& ax = point_pairs[0]; ax.resize(input.size());
    Eigen::ArrayXf& ay = point_pairs[1]; ay.resize(input.size());
    Eigen::ArrayXf& bx = point_pairs[2]; bx.resize(input.size());
    Eigen::ArrayXf& by = point_pairs[3]; by.resize(input.size());
    for(size_t i = 0; i <input.size(); i++)
    {
        const auto& match = (*mMatches)[input[i]];
        const cv::Point2f& a = (*mKpts0)[match.queryIdx].pt;
        ax[i] = a.x;
        ay[i] = a.y;
        const cv::Point2f& b = (*mKpts1)[match.trainIdx].pt;
        bx[i] = b.x;
        by[i] = b.y;
    }

    auto indices2pts = [&](const std::array<uint32_t, 3>& indices) -> Eigen::Matrix<float, 3, 4>{
        Eigen::Matrix<float, 3, 4> pts;
        for(size_t i = 0; i < indices.size(); i++)
        {
            const uint32_t idx = indices[i];
            pts.row(i) <<  ax[idx], ay[idx], bx[idx], by[idx];
        }
        return pts;
    };

    auto ransac_test = [&](const std::array<uint32_t, 3>& indices) -> uint32_t {
        Eigen::Matrix<float, 3, 4> pts = indices2pts(indices);
#if 1
        affine2_t<float> trans = find_affine2(pts);
#else
        affine2_t<float> trans = find_sim2(pts.template topRows<2>());
#endif
        float scale = trans.matrix().template leftCols<2>().determinant();
        if(scale < 0.25f || scale > 4.f)
            return 0u;

        return (uint32_t)check_transform_affine2(trans, point_pairs, mCellWidth * threshold_relative).count();
    };

    std::array<uint32_t, 3> best_indices = ransac<decltype(ransac_test), 3, uint32_t>(ransac_test, input.size(), ransac_confidence, ransac_max_iterations);
    const affine2_t<float> trans_ransac = find_affine2(indices2pts(best_indices));
    affine2_t<float> trans_neighbour0 = affine2_t<float>::Identity();
    affine2_t<float> trans_neighbour1 = affine2_t<float>::Identity();
    if(direction == Direction::Forward){
        if(x > 0)
            trans_neighbour0 = param(y, x-1).trans;
        if(y > 0)
            trans_neighbour1 = param(y-1, x).trans;
    }
    else
    {
        if(x < mCols - 1)
            trans_neighbour0 = param(y, x + 1).trans;
        if(y < mRows - 1)
            trans_neighbour1 = param(y + 1, x).trans;
    }
    Eigen::Array<bool, Eigen::Dynamic, 1> mask;
    uint32_t numInliers = 0;
    for(auto trans : {trans_ransac, trans_neighbour0, trans_neighbour1})
    {
        mask = check_transform_affine2(trans, point_pairs, mCellWidth * threshold_relative);
        numInliers = uint32_t(mask.count());
        //refine trans
        {
            Eigen::Matrix<float, -1, 4> pts(numInliers, 4);
            uint32_t p = 0;
            for(size_t i = 0; i < input.size(); i++)
            {
                if(mask[i])
                    pts.row(p++) <<  ax[i], ay[i], bx[i], by[i];
            }
            assert(p == numInliers);
            trans = find_affine2(pts);
            mask = check_transform_affine2(trans, point_pairs, mCellWidth * threshold_relative);
            numInliers = uint32_t(mask.count());
        }
        if(numInliers > param(y,x).numInliers)
            param(y, x) = {trans, numInliers};
    }
    if(vote && int(numInliers) > 4){
        for(unsigned i = 0; i < input.size(); i++){
            if(mask[i])
                mVotes[input[i]]++;
        }
    }
}

void PropagationMatchFilter::propagate(Direction direction, bool vote)
{
    if(direction == Direction::Forward) {
        for (uint32_t i = 0; i < mRows; i++) {
            for (uint32_t j = 0; j < mCols; j++) {
                processCell(i, j, direction, vote);
            }
        }
    }else{
        for (int i = mRows-1; i >= 0; i--) {
            for (int j = mCols-1; j >= 0; j--) {
                processCell(uint32_t(i), uint32_t(j), direction, vote);
            }
        }
    }
}

void PropagationMatchFilter::run()
{
    propagate(Direction::Forward, false);
    propagate(Direction::Backward, true);
}

std::vector<bool> PropagationMatchFilter::getInlierMask(int minVotes) const
{
    std::vector<bool> mask(mMatches->size());
    for(unsigned i = 0; i < mMatches->size(); i++)
        mask[i] = mVotes[i] > minVotes;
    return mask;
}
}