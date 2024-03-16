//
// Created by yao on 21/08/17.
//
#include "geometry.h"

Eigen::Array<bool, Eigen::Dynamic, 1> check_transform_affine2(
        const affine2_t<float> &trans, const Eigen::ArrayXf *point_pairs, const float threshold)
{
    const auto& mat = trans.matrix();

    const float scale = std::sqrt(std::abs(trans.linear().determinant()));
    const float sqr_threshold = (threshold * std::min(scale, 1.f)) * (threshold * std::min(scale, 1.f));

    return (mat(0,2) + mat(0,0) * point_pairs[0] + mat(0,1) * point_pairs[1] - point_pairs[2]).square()
           +(mat(1,2) + mat(1,0) * point_pairs[0] + mat(1,1) * point_pairs[1] - point_pairs[3]).square() < sqr_threshold;
}

Eigen::Array<bool, Eigen::Dynamic, 1> check_transform_projective(
        const Eigen::Projective2f& trans, const Eigen::ArrayXf point_pairs[4], const float threshold)
{
    const auto& mat = trans.matrix();

    const Eigen::ArrayXf z = (mat(2, 2) + mat(2, 0) * point_pairs[0] + mat(2, 1) * point_pairs[1]);

    const float sqr_threshold = threshold*threshold;
    const Eigen::ArrayXf sqr_scale = z.square().min(std::abs(trans.linear().determinant()));

    return ((mat(0,2) + mat(0,0) * point_pairs[0] + mat(0,1) * point_pairs[1]) - point_pairs[2] * z).square()
           + ((mat(1,2) + mat(1,0) * point_pairs[0] + mat(1,1) * point_pairs[1]) - point_pairs[3] * z).square() < sqr_threshold * sqr_scale;
}

Eigen::Array<bool, Eigen::Dynamic, 1> check_transform_epipolarity(
        const Eigen::Matrix3f& F, const Eigen::ArrayXf point_pairs[4], const float threshold){
    const auto& x0 = point_pairs[0];
    const auto& y0 = point_pairs[1];
    const auto& x1 = point_pairs[2];
    const auto& y1 = point_pairs[3];

    //@todo: do one of the following:
    // 1. if number of points is high, use a for loop with each iteration calculating ~1024 points (fit in 64K L1 cache per core).
    // 2. optimise by using point_pairs as padded Eigen::ArrayX4f. check Eigen to see if that results in aligned load
    const Eigen::ArrayXf epipole0[3] = {
            x1 * F(0, 0) + (y1 * F(1, 0) + F(2, 0)),
            x1 * F(0, 1) + (y1 * F(1, 1) + F(2, 1)),
            x1 * F(0, 2) + (y1 * F(1, 2) + F(2, 2))
    };
    const Eigen::ArrayXf epipole1[2] = {
            x0 * F(0, 0) + (y0 * F(0, 1) + F(0, 2)),
            x0 * F(1, 0) + (y0 * F(1, 1) + F(1, 2))//,
            //x0 * F(2, 0) + (y0 * F(2, 1) + F(2, 2)) //not used
    };

    //(X1' * F * X0).squared()
    const Eigen::ArrayXf squared_X1_FX0 = (epipole0[0] * x0 + (epipole0[1] * y0 + epipole0[2])).square();

    const float squared_threshold = threshold * threshold;

    Eigen::Array<bool, Eigen::Dynamic, 1> mask = (squared_X1_FX0 < squared_threshold * (epipole0[0].square() + epipole0[1].square()))
                                                 && (squared_X1_FX0 < squared_threshold * (epipole1[0].square() + epipole1[1].square()));

    return mask;
};