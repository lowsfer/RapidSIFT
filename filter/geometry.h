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
// Created by yao on 21/08/17.
//

#pragma once
#include <eigen3/Eigen/Eigen>
#include "types.h"
#include "utils_sfm.h"
#include <eigen3/Eigen/StdVector>

template<typename Derived>
isometry2_t<float> find_sim2(const Eigen::MatrixBase<Derived>& pts)
{
    Eigen::Matrix<float, Derived::RowsAtCompileTime < 0 ? -1 : Derived::RowsAtCompileTime * 2, Derived::ColsAtCompileTime> A(pts.rows() * 2, 4);
    Eigen::Matrix<float, Derived::RowsAtCompileTime < 0 ? -1 : Derived::RowsAtCompileTime * 2, 1> b(pts.rows() * 2, 1);

    //@todo: use vector operations to fill A and b instead of for loops
    for(int i = 0; i < pts.rows(); i++)
    {
        A.template block<2, 4>(i * 2, 0)
                << pts(i, 0), -pts(i, 1), 1, 0,
                pts(i, 1), pts(i, 0), 0, 1;
        b.template block<2, 1>(i * 2, 0)
                << pts(i, 2), pts(i, 3);
    }

    Eigen::Vector4f X = least_square_solve(A, b);

//    A <<    pts(0, 0), -pts(0, 1), 1, 0,
//            pts(0, 1), pts(0, 0), 0, 1,
//            pts(1, 0), -pts(1, 1), 1, 0,
//            pts(1, 1), pts(1, 0), 0, 1;
//    b <<    pts(0, 2), pts(0, 3),
//            pts(1, 2), pts(1, 3);
    //Eigen::Vector4d X = A.colPivHouseholderQr().solve(b);

    isometry2_t<float> result = isometry2_t<float>::Identity();
    result.matrix().topRows<2>() << X[0], -X[1], X[2],
            X[1], X[0], X[3];

    return result;
}

template<typename Derived>
affine2_t<float> find_affine2(const Eigen::MatrixBase<Derived>& pts)
{
    Eigen::Matrix<float, Derived::RowsAtCompileTime, 3> A = pts.template leftCols<2>().rowwise().homogeneous();
    Eigen::Matrix<float, Derived::RowsAtCompileTime, 2> b = pts.template rightCols<2>();

    affine2_t<float> result;

    if(pts.rows() == 3)
        result.matrix().transpose() = A.inverse() * b;
    else
        result.matrix().transpose() = least_square_solve(A, b);

    return result;
}

Eigen::Array<bool, Eigen::Dynamic, 1> check_transform_affine2(
        const affine2_t<float> &trans, const Eigen::ArrayXf *point_pairs, const float threshold);

template<typename Derived>
std::tuple<affine2_t<typename Derived::Scalar>, affine2_t<typename Derived::Scalar>, Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>>
    hartley_normalise(const Eigen::MatrixBase<Derived>& pts)
{
    if(pts.cols() != 4)
        throw std::runtime_error("should have 4 columns: ax, ay, bx, by");
    typedef typename Derived::Scalar scalar_type;
    Eigen::Matrix<scalar_type, 2, 4> bound;
    bound.row(0) = pts.colwise().minCoeff();
    bound.row(1) = pts.colwise().maxCoeff();
    Eigen::Matrix<scalar_type, 1, 4> centre = pts.colwise().mean();
    Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> pts_norm(pts.rows(), pts.cols());
    pts_norm << pts.template leftCols<2>().rowwise() - centre.template leftCols<2>(), pts.template rightCols<2>().rowwise() - centre.template rightCols<2>();

    const scalar_type scale[2] = {
            std::sqrt(2.f) / pts_norm.template leftCols<2>().rowwise().norm().mean(),
            std::sqrt(2.f) / pts_norm.template rightCols<2>().rowwise().norm().mean()
    };

    affine2_t<scalar_type> trans[2];
    trans[0].matrix() << scale[0], 0, scale[0] * -centre[0],
            0, scale[0], scale[0] * -centre[1];
    trans[1].matrix() << scale[1], 0, scale[1] * -centre[2],
            0, scale[1], scale[1] * -centre[3];


    pts_norm.template leftCols<2>() *= scale[0];
    pts_norm.template rightCols<2>() *= scale[1];

    return std::make_tuple(trans[0], trans[1], pts_norm);
}


template<typename Derived, bool PreferAccuracy = true>
Eigen::Transform<typename Derived::Scalar, 2, Eigen::Projective> find_homography(const Eigen::MatrixBase<Derived>& pts)
{
    if(pts.rows() < 4)
        throw std::runtime_error("at least 4 points are required");
    typedef typename Derived::Scalar scalar_type;
    affine2_t<scalar_type> trans[2];
    Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> pts_norm;
    std::tie(trans[0], trans[1], pts_norm) = hartley_normalise(pts);
    //trans[0].setIdentity(); trans[1].setIdentity(); pts_norm = pts;
    Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime < 0 ? -1 : Derived::RowsAtCompileTime * 2, 9> P(pts.rows() * 2, 9);

//    auto x = pts_norm.col(0).array().eval();
//    auto y = pts_norm.col(1).array().eval();
//    auto x_ = pts_norm.col(2).array().eval();
//    auto y_ = pts_norm.col(3).array().eval();
//    typedef decltype(x) vec_type;
//    P << -x, -y, -vec_type::Ones(pts.rows()), vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), x*x_, y*x_, x_,
//            vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), vec_type::Zero(pts.rows()), -x, -y, -vec_type::Ones(), x*y_, y*y_, y_;

#define x0_y0 pts_norm.array().template leftCols<2>()
#define x1 pts_norm.array().col(2)
#define y1 pts_norm.array().col(3)
#define vec_ones Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, 1>::Ones(pts.rows())
    P << -x0_y0, -vec_ones,
            Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, 3>::Zero(pts.rows(), 3),
            x0_y0.colwise() * x1, x1,
            Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, 3>::Zero(pts.rows(), 3),
            -x0_y0, -vec_ones,
            x0_y0.colwise() * y1, y1;
#undef vec_ones
#undef x1
#undef y1
#undef x0_y0
//    //@todo: use vector operations to fill P instead of for loops - done
//    for(int i = 0; i < pts.rows(); i++)
//    {
//        const scalar_type x = pts_norm(i, 0);
//        const scalar_type y = pts_norm(i, 1);
//        const scalar_type x_ = pts_norm(i, 2);
//        const scalar_type y_ = pts_norm(i, 3);
//        P.template middleRows<2>(i * 2)
//                << -x, -y, -1.f, 0.f, 0.f, 0.f, x*x_, y*x_, x_,
//                0.f, 0.f, 0.f, -x, -y, -1, x*y_, y*y_, y_;
//    }

    Eigen::Matrix<scalar_type, 9, 1> h_norm;
    if(PreferAccuracy){
        //more accurate
        auto svd = P.jacobiSvd(Eigen::ComputeFullV);
        h_norm = svd.matrixV().col(8);
    }else {
        //less accurate, but faster
        auto svd = (P.transpose() * P).eval().jacobiSvd(Eigen::ComputeFullV);
        h_norm = svd.matrixV().col(8);
    }

    Eigen::Matrix<scalar_type, 3, 3> st[2];
    for(int i = 0; i < 2; i++)
        st[i] << trans[i].matrix(),
                0, 0, 1;
    Eigen::Transform<scalar_type, 2, Eigen::Projective> H;
    H.matrix() = st[1].inverse() * Eigen::Matrix<scalar_type, 3,3, Eigen::RowMajor>::Map(h_norm.data()) * st[0];
    return H;
}

Eigen::Array<bool, Eigen::Dynamic, 1> check_transform_projective(
        const Eigen::Projective2f& trans, const Eigen::ArrayXf point_pairs[4], const float threshold);

template<typename Derived, bool PreferAccuracy = true>
Eigen::Matrix<typename Derived::Scalar, 3, 3> find_epipolarity(const Eigen::MatrixBase<Derived>& pts){
    if(pts.rows() < 8)
        throw std::runtime_error("at least 8 points are required");
    typedef typename Derived::Scalar scalar_type;
    affine2_t<scalar_type> trans[2];
    Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> pts_norm;
    std::tie(trans[0], trans[1], pts_norm) = hartley_normalise(pts);
    Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, 9> P(pts.rows(), 9);

    //e0*x*x_ + e1*x*y_ + e2*x + e3*x_*y + e4*y*y_ + e5*y + e6*x_ + e7*y_ + e8, e in col-major 3x3 matrix
#define x0 pts_norm.array().col(0)
#define y0 pts_norm.array().col(1)
#define x1_y1 pts_norm.array().template rightCols<2>()
#define vec_ones Eigen::Matrix<scalar_type, Derived::RowsAtCompileTime, 1>::Ones(pts.rows())
    P << x1_y1.colwise() * x0, x0, x1_y1.colwise() * y0, y0, x1_y1, vec_ones;
#undef vec_ones
#undef x0
#undef y0
#undef x1_y1

    Eigen::Matrix<scalar_type, 9, 1> f_norm;
    if(PreferAccuracy){
        //more accurate
        auto svd = P.jacobiSvd(Eigen::ComputeFullV);
        f_norm = svd.matrixV().col(8);
    }else {
        //less accurate, but faster
        auto svd = (P.transpose() * P).eval().jacobiSvd(Eigen::ComputeFullV);
        f_norm = svd.matrixV().col(8);
    }

    Eigen::Matrix<scalar_type, 3, 3> st[2];
    for(int i = 0; i < 2; i++)
        st[i] << trans[i].matrix(),
                0, 0, 1;
    Eigen::Matrix<scalar_type, 3, 3> F_est = Eigen::Matrix<scalar_type, 3,3>::Map(f_norm.data());

    auto svd_f33 = F_est.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<scalar_type, 3, 1> S = svd_f33.singularValues();
    S[2] = 0;
    Eigen::Matrix<scalar_type, 3, 3> F = st[1].transpose() * svd_f33.matrixU() * S.asDiagonal() * svd_f33.matrixV().transpose() * st[0];

    return F;
}

Eigen::Array<bool, Eigen::Dynamic, 1> check_transform_epipolarity(
        const Eigen::Matrix3f& F, const Eigen::ArrayXf point_pairs[4], const float threshold);

template<typename T, typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 3> triangulate(
        const camera_t<T> &camera0, const camera_t<T> &camera1, const Isometry3<T> &RT,
        const Eigen::MatrixBase<Derived> &pairs);// __attribute__((optimize ("fast-math")));

//@todo: add unit test
template<typename T, typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 3> triangulate(
        const camera_t<T> &camera0, const camera_t<T> &camera1, const Isometry3<T> &RT,
        const Eigen::MatrixBase<Derived> &pairs)
{
    typedef typename Derived::Scalar scalar_t;
    const int num_pairs = pairs.rows();
    assert(pairs.cols() == 4);
    //use Eigen::Array as scalar type of to solve all vertex pairs in a batch?
    const Eigen::Matrix<scalar_t, Derived::RowsAtCompileTime, 1>
            X0 = pairs.col(0), Y0 = pairs.col(1),
            X1 = pairs.col(2), Y1 = pairs.col(3);
    Eigen::Matrix<scalar_t, Derived::RowsAtCompileTime, 1> vec_x(num_pairs, 1), vec_y(num_pairs, 1), vec_z(num_pairs, 1);
    Eigen::Matrix<scalar_t, 3, 4> P = (camera1.K()*RT.matrix().template topRows<3>()).template cast<scalar_t>();
    const scalar_t P00 = P(0, 0), P01 = P(0, 1), P02 = P(0, 2), P03 = P(0, 3), P10 = P(1, 0), P11 = P(1, 1), P12 = P(1, 2), P13 = P(1, 3), P20 = P(2, 0), P21 = P(2, 1), P22 = P(2, 2), P23 = P(2, 3);
    const scalar_t fx0 = camera0.fx(), fy0 = camera0.fy(), cx0 = camera0.cx(), cy0 = camera0.cy();
    //factors for A[3][3]
    const scalar_t CA00[5] = { P20*P20, P00*P00 + P10*P10 + fx0*fx0, -2*P00*P20, -2*P10*P20, P20*P20};
    const scalar_t CA01[5] = { P20*P21, P00*P01 + P10*P11, -P00*P21 - P01*P20, -P10*P21 - P11*P20, P20*P21};
    const scalar_t CA02[6] = { P00*P02 + P10*P12 + cx0*fx0, -P00*P22 - P02*P20, P20*P22, -P10*P22 - P12*P20, P20*P22, -fx0};
    const scalar_t CA10[5] = { P20*P21, P00*P01 + P10*P11, -P00*P21 - P01*P20, -P10*P21 - P11*P20, P20*P21};
    const scalar_t CA11[5] = { P21*P21, P01*P01 + P11*P11 + fy0*fy0, -2*P01*P21, -2*P11*P21, P21*P21};
    const scalar_t CA12[6] = { P01*P02 + P11*P12 + cy0*fy0, -P01*P22 - P02*P21, -fy0, P21*P22, -P11*P22 - P12*P21, P21*P22};
    const scalar_t CA20[6] = { P00*P02 + P10*P12 + cx0*fx0, -P00*P22 - P02*P20, P20*P22, -P10*P22 - P12*P20, P20*P22, -fx0};
    const scalar_t CA21[6] = { P01*P02 + P11*P12 + cy0*fy0, -P01*P22 - P02*P21, -fy0, P21*P22, -P11*P22 - P12*P21, P21*P22};
    const scalar_t CA22[9] = { P02*P02 + P12*P12 + cx0*cx0 + cy0*cy0, -2*P02*P22, 1, P22*P22, 1, -2*P12*P22, -2*cx0, P22*P22, -2*cy0};
    const scalar_t Cb0[5] = { -P20*P23, -P00*P03 - P10*P13, P00*P23 + P03*P20, P10*P23 + P13*P20, -P20*P23};
    const scalar_t Cb1[5] = { -P21*P23, -P01*P03 - P11*P13, P01*P23 + P03*P21, P11*P23 + P13*P21, -P21*P23};
    const scalar_t Cb2[5] = { -P22*P23, -P02*P03 - P12*P13, P02*P23 + P03*P22, P12*P23 + P13*P22, -P22*P23};
    vectorize_loop
    for(int i = 0; i < num_pairs; i++)
    {
        const scalar_t x0 = X0[i], y0 = Y0[i], x1 = X1[i], y1 = Y1[i];
        const scalar_t term[8] = {x0*x0, y0*y0, x1*x1, y1*y1, x0, y0, x1, y1};
        const scalar_t A00 = term[2] * CA00[0] + CA00[1] + term[6] * CA00[2] + term[7] * CA00[3] + term[3] * CA00[4];
        const scalar_t A01 = term[2] * CA01[0] + CA01[1] + term[6] * CA01[2] + term[7] * CA01[3] + term[3] * CA01[4];
        const scalar_t A02 = CA02[0] + term[6] * CA02[1] + term[2] * CA02[2] + term[7] * CA02[3] + term[3] * CA02[4] + term[4] * CA02[5];
        const scalar_t A10 = term[2] * CA10[0] + CA10[1] + term[6] * CA10[2] + term[7] * CA10[3] + term[3] * CA10[4];
        const scalar_t A11 = term[2] * CA11[0] + CA11[1] + term[6] * CA11[2] + term[7] * CA11[3] + term[3] * CA11[4];
        const scalar_t A12 = CA12[0] + term[6] * CA12[1] + term[5] * CA12[2] + term[2] * CA12[3] + term[7] * CA12[4] + term[3] * CA12[5];
        const scalar_t A20 = CA20[0] + term[6] * CA20[1] + term[2] * CA20[2] + term[7] * CA20[3] + term[3] * CA20[4] + term[4] * CA20[5];
        const scalar_t A21 = CA21[0] + term[6] * CA21[1] + term[5] * CA21[2] + term[2] * CA21[3] + term[7] * CA21[4] + term[3] * CA21[5];
        const scalar_t A22 = CA22[0] + term[6] * CA22[1] + term[1] * CA22[2] + term[2] * CA22[3] + term[0] * CA22[4] + term[7] * CA22[5] + term[4] * CA22[6] + term[3] * CA22[7] + term[5] * CA22[8];
        const scalar_t b0 = term[2] * Cb0[0] + Cb0[1] + term[6] * Cb0[2] + term[7] * Cb0[3] + term[3] * Cb0[4];
        const scalar_t b1 = term[2] * Cb1[0] + Cb1[1] + term[6] * Cb1[2] + term[7] * Cb1[3] + term[3] * Cb1[4];
        const scalar_t b2 = term[2] * Cb2[0] + Cb2[1] + term[6] * Cb2[2] + term[7] * Cb2[3] + term[3] * Cb2[4];

        const scalar_t A11A22_A12A21 = A11*A22 - A12*A21;
        const scalar_t A10A22 = A10*A22, A12A20 = A12*A20, A10A21 = A10*A21, A11A20 = A11*A20;
        const scalar_t x = b0*(A11A22_A12A21) + b1*(-A01*A22 + A02*A21) + b2*(A01*A12 - A02*A11);
        const scalar_t y = -b0*(A10A22 - A12A20) - b1*(-A00*A22 + A02*A20) - b2*(A00*A12 - A02*A10);
        const scalar_t z = b0*(A10A21 - A11A20) + b1*(-A00*A21 + A01*A20) + b2*(A00*A11 - A01*A10);
        const scalar_t scale = 1.0 / (A00*(A11A22_A12A21) - A01*(A10A22 - A12A20) + A02*(A10A21 - A11A20));
        vec_x[i] = x * scale; vec_y[i] = y * scale; vec_z[i] = z * scale;
    }
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 3> result(num_pairs, 3);
    result << vec_x, vec_y, vec_z;

    const Eigen::Matrix<bool, Derived::RowsAtCompileTime, 1> sanity = result.array().isFinite().rowwise().all();
    const scalar_t fx0_inv = 1.f / fx0;
    const scalar_t fy0_inv = 1.f / fy0;
    vectorize_loop
    for(int i = 0; i < num_pairs; i++){
        if(std::abs(result(i, 2)) < 1E-10f)
            result(i, 2) = 1E-3f;
        if(!sanity[i]){
            const Vector2<scalar_t> pt2d = pairs.template block<1, 2>(i, 0).transpose();
            Vector3<scalar_t> pt3d;
            pt3d << (pt2d[0] - cx0) * fx0_inv, (pt2d[1] - cy0) * fy0_inv, 1.f;
            result.row(i) = scalar_t(1000.f) * pt3d.transpose();
        }
    }

    return result;
};

template <typename scalar_type>
std::array<Isometry3<scalar_type>, 4> decompose_essential_matrix(
        const Matrix3<scalar_type>& E, Vector3<scalar_type>* singular_values = nullptr)
{
    /*https://en.wikipedia.org/wiki/Essential_matrix*/
    Eigen::JacobiSVD<Matrix3<scalar_type>> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Matrix3<scalar_type> U = svd.matrixU();
    Matrix3<scalar_type> V = svd.matrixV();
    //Vector3<scalar_type> S = svd.singularValues();
    if(U.determinant() < 0)
    {
        U.col(2) = -U.col(2);
        //S[2] = -S[2];
    }
    if(V.determinant() < 0)
    {
        V.col(2) = -V.col(2);
        //S[2] = -S[2];
    }

    if(singular_values != nullptr)
        *singular_values = svd.singularValues();
    Matrix3<scalar_type> W, Z;
    W <<    0, -1, 0,
            1, 0, 0,
            0, 0, 1;
    Z <<    0, 1, 0,
            -1, 0, 0,
            0, 0, 0;
    Matrix3<scalar_type> R[2] = {
            U * W.transpose() * V.transpose(),
            U * W * V.transpose()
    };
    assert(R[0].determinant() > 0 && R[1].determinant() > 0);

    //const Matrix3<scalar_type> Tx = U * W * S.asDiagonal() * U.transpose();
    const Matrix3<scalar_type> Tx = U * Z * U.transpose();
    Vector3<scalar_type> T[2];
    T[0] << cross_mat2vec(Tx).normalized();
    T[1] = -T[0];

    std::array<Isometry3<scalar_type>, 4> solution_list;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            auto& solution = solution_list[i*2 + j];
            solution.setIdentity();
            solution.linear() = R[i];
            solution.translation() = T[j];
        }
    }
    return solution_list;
};

template <typename scalar_type>
std::array<Isometry3<scalar_type>, 4> decompose_epipolarity(
        const camera_t<scalar_type>& cam0, const camera_t<scalar_type>& cam1,
        const Matrix3<scalar_type>& F, Vector3<scalar_type>* singular_values = nullptr)
{
    /*https://en.wikipedia.org/wiki/Essential_matrix*/
    const Matrix3<scalar_type> E = cam1.K().transpose() * F * cam0.K();
    return decompose_essential_matrix(E, singular_values);
};

template<typename T>
struct homography_decomposed_t{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Matrix3<T> R;
    Vector3<T> N;
    Vector3<T> T_by_d;// 1/d * T

    //T is normalized in returned RT
    Isometry3<T> get_RT() const{
        Isometry3<T> RT(R);
        RT.translation() = T_by_d.normalized();
        return RT;
    }
};

//decompose_H_F.pdf
template<typename scalar>
std::vector<homography_decomposed_t<scalar>> decompose_homography(const Matrix3<scalar>& H)
{
    //normalise
    scalar scale = H.jacobiSvd().singularValues()[1];
    Matrix3<scalar> H_norm = 1 / scale * H;

    auto svd = (H_norm.transpose() * H_norm).eval().jacobiSvd(Eigen::ComputeFullV);
    const Vector3<scalar>& S = svd.singularValues();
    const Matrix3<scalar> V = svd.matrixV().determinant() > 0 ? svd.matrixV() : (-svd.matrixV()).eval();
    assert(V.determinant() > 0);
    const Vector3<scalar> v1 = V.col(0);
    const Vector3<scalar> v2 = V.col(1);
    const Vector3<scalar> v3 = V.col(2);

    scalar c = 1 / std::sqrt(S[0] - S[2]);
    scalar a = std::sqrt(1 - S[2]) * c;
    scalar b = std::sqrt(S[0] - 1) * c;

    const Vector3<scalar> u1 = a * v1 + b * v3;
    const Vector3<scalar> u2 = a * v1 - b * v3;

    Matrix3<scalar> U1, U2, W1, W2;
    U1 << v2, u1, v2.cross(u1);
    U2 << v2, u2, v2.cross(u2);

    std::array<homography_decomposed_t<scalar>, 8> solution_list;
    for(int i = 0; i < 2; i++)
    {
        if(i == 1)
            H_norm = -H_norm;//negative of homography matrix is the same transformation

        W1 << H_norm * v2, H_norm * u1, (H_norm * v2).cross(H_norm * u1);
        W2 << H_norm * v2, H_norm * u2, (H_norm * v2).cross(H_norm * u2);
        {
            Matrix3<scalar> R = W1 * U1.transpose();
            Vector3<scalar> N = v2.cross(u1);
            Vector3<scalar> T_by_d = (H_norm - R) * N;
            solution_list[i*4] = {R, N, T_by_d};
        }
        {
            Matrix3<scalar> R = W2 * U2.transpose();
            Vector3<scalar> N = v2.cross(u2);
            Vector3<scalar> T_by_d = (H_norm - R) * N;
            solution_list[i*4 + 1] = {R, N, T_by_d};
        }
        {
            Matrix3<scalar> R = solution_list[i*4].R;
            Vector3<scalar> N = -solution_list[i*4].N;
            Vector3<scalar> T_by_d = -solution_list[i*4].T_by_d;
            solution_list[i*4 + 2] = {R, N, T_by_d};
        }
        {
            Matrix3<scalar> R = solution_list[i*4 + 1].R;
            Vector3<scalar> N = -solution_list[i*4 + 1].N;
            Vector3<scalar> T_by_d = -solution_list[i*4 + 1].T_by_d;
            solution_list[i*4 + 3] = {R, N, T_by_d};;
        }
    }
    std::array<bool, 8> mask;
    for(int i = 0; i < 8; i++) {
        mask[i] = solution_list[i].N.dot(solution_list[i].T_by_d) < 1;
    }

    std::vector<homography_decomposed_t<scalar>> valid_solutions;
    valid_solutions.reserve(8);
    for(unsigned i = 0; i < solution_list.size(); i++)
        if(mask[i])
            valid_solutions.emplace_back(solution_list[i]);

    return valid_solutions;
};

template<typename scalar>
std::vector<homography_decomposed_t<scalar>> decompose_homography(
        const camera_t<scalar>& cam0, const camera_t<scalar>& cam1,
        const Matrix3<scalar>& G
)
{
    //normalise
    Matrix3<scalar> H = cam1.K().inverse() * G * cam0.K();
    return decompose_homography(H);
};

#include <opencv2/calib3d.hpp>
template<typename scalar>
std::vector<homography_decomposed_t<scalar>> decompose_homography_opencv(
        const camera_t<scalar>& cam0, const camera_t<scalar>& cam1,
        const Matrix3<scalar>& G
)
{
    UNUSED(cam1);
    std::vector<cv::Mat> R_list, T_list, N_list;
    cv::Mat H(3, 3, CV_32F);
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            H.at<float>(i,j) = G(i,j);
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0,0) = cam0.fx();
    K.at<float>(1,1) = cam0.fy();
    K.at<float>(0,2) = cam0.cx();
    K.at<float>(1,2) = cam0.cy();
    cv::decomposeHomographyMat(H, K, R_list, T_list, N_list);
    std::vector<homography_decomposed_t<scalar>> result;
    for(unsigned n = 0; n < R_list.size(); n++){
        homography_decomposed_t<scalar> solution;
        for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
                solution.R(i,j) = R_list[n].at<float>(i,j);
        for(int i = 0; i < 3; i++)
            solution.N[i] = N_list[n].at<float>(i);
        for(int i = 0; i < 3; i++)
            solution.T_by_d[i] = T_list[n].at<float>(i);
        result.push_back(solution);
    }
    return result;
};

//triangulate a single point from multiple lines
template<typename Scalar>
vec3<Scalar> triangulate(std::vector<std::tuple<const Isometry3<Scalar>*, const camera_t<Scalar>*, vec2<Scalar>>> observ_list);

//@todo: add unit test
//https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
template<typename Scalar>
vec3<Scalar> triangulate(std::vector<std::tuple<const Isometry3<Scalar>*, const camera_t<Scalar>*, vec2<Scalar>>> observ_list){
    Matrix3<Scalar> A = Matrix3<Scalar>::Zero();
    Vector3<Scalar> b = Vector3<Scalar>::Zero();
    for(const auto& ob : observ_list){
        const Isometry3<Scalar>& RT = *std::get<0>(ob);
        const camera_t<Scalar>& camera = *std::get<1>(ob);
        const vec2<Scalar>& pt2d = std::get<2>(ob);

        const Vector3<Scalar> p = RT.linear().transpose() * -RT.translation();

        Vector3<Scalar> v;
        v << (pt2d.x - camera.cx()) / camera.fx(), (pt2d.y - camera.cy()) / camera.fy(), 1.f;
        v = RT.linear().transpose() * v;
        v.normalize();

        const Matrix3<Scalar> M = Matrix3<Scalar>::Identity() - v * v.transpose();
        A += M;
        b += M * p;
    }
    vec3<Scalar> pt3d;
    pt3d.map() = A.inverse() * b;
    return pt3d;
}