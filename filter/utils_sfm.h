#pragma once
#include <eigen3/Eigen/Dense>
#include <exception>
#include "types.h"
#include <boost/pending/disjoint_sets.hpp>
#include <map>
//#include <set>
#include <boost/predef.h>

#define require(expr)	do{if(!(expr)) std::terminate();}while(0)

#ifdef __GNUC__
    #ifndef __clang__
        #define vectorize_loop _Pragma("GCC ivdep")
    #endif
#endif
#ifdef __clang__
    #define vectorize_loop _Pragma("clang loop vectorize(enable) interleave(enable)")
#endif


#define strcat_(x, y) x ## y
#define strcat(x, y) strcat_(x, y)
#define static_print_value(x) template <int> struct strcat(strcat(value_of_, x), _is); static_assert(strcat(strcat(value_of_, x), _is)<x>::x, "");


template<typename T>
inline T square(T val){
    return val*val;
}

template<typename T>
inline T cube(T val){
    return val*val*val;
}

#define UNUSED(x) (void)(x)

enum class least_square_solver_type{
    SVD, QR, NormalEquation
};

template<typename Derived1, typename Derived2, least_square_solver_type solver_type = least_square_solver_type::NormalEquation>
Eigen::Matrix<typename Derived1::Scalar, Derived1::ColsAtCompileTime, Derived2::ColsAtCompileTime>
    least_square_solve(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& b)
{
    Eigen::Matrix<typename Derived1::Scalar, Derived1::ColsAtCompileTime, Derived2::ColsAtCompileTime> X(A.cols(), b.cols());

    if(A.rows() == A.cols() && A.rows() < 4)
    {
        X = A.inverse() * b;
    }
    else{
        switch(solver_type)
        {
        case least_square_solver_type::SVD:
            X = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b).eval();
            break;
        case least_square_solver_type::QR:
            X = A.householderQr().solve(b).eval();
            break;
        case least_square_solver_type::NormalEquation:
            X = (A.transpose() * A).llt().solve(A.transpose() * b).eval();
            break;
        default:
            throw std::runtime_error("fatal error");
        }
    }
    return X;
}

template<typename T>
constexpr inline const T& clamp(const T& x, const T& lb, const T& ub){
    return std::max(std::min(x, ub), lb);
}

// pixels_at_gird_cross_points defines coordinate style of pt. true means OpenGL/CUDA/D3D texture style.
// In this program we always use false.
template<bool pixels_at_gird_cross_points = false>
cv::Vec3b sample_image(const cv::Mat &img, const cv::Point2f &pt)
{
    assert(!img.empty());
    assert(img.type() == CV_8UC3);

    float x[2];
    float y[2];
    for(int i = 0; i < 2; i++)
    {
        x[i] = clamp(static_cast<int>(std::floor(pt.x)) + i, 0, img.cols - 1);
        y[i] = clamp(static_cast<int>(std::floor(pt.y)) + i, 0, img.rows - 1);
    }

    cv::Vec3b corners[2][2];

    for(int i = 0; i < 2; i++)
        for(int j = 0; j < 2; j++) {
            if(pixels_at_gird_cross_points)
                corners[i][j] = img.at<cv::Vec3b>(y[j] - 0.5f, x[i] - 0.5f);
            else
                corners[i][j] = img.at<cv::Vec3b>(y[j], x[i]);
        }


    float x0 = pt.x - x[0], x1 = x[1] - pt.x;
    float y0 = pt.y - y[0], y1 = y[1] - pt.y;
    float scale = 1.f / ((x0 + x1) * (y0 + y1));
    float weight[2][2] = {
            {x1 * y1 * scale, x1 * y0 * scale},
            {x0 * y1 * scale, x0 * y0 * scale}
    };

    cv::Vec3b result;
    for(int c = 0; c < 3; c++)
    {
        result(c) = static_cast<uchar>(clamp(static_cast<int>(round(corners[0][0](c) * weight[0][0]
                                                                    + corners[0][1](c) * weight[0][1]
                                                                    + corners[1][0](c) * weight[1][0]
                                                                    + corners[1][1](c) * weight[1][1])), 0, 255));
    }

    return result;
}

template<typename T>
Matrix3<T> gvec2mat(const T gvec[3])
{
    const T& g0 = gvec[0];
    const T& g1 = gvec[1];
    const T& g2 = gvec[2];
    const T factor = 2 / (g0*g0 + g1*g1 + g2*g2 + 1);
    Eigen::Matrix<T, 3, 3> R;
    R <<    g0*g0 + 1,   g0*g1 + g2, g0*g2 - g1,
            g0*g1 - g2,  g1*g1 + 1,  g1*g2 + g0,
            g0*g2 + g1,  g1*g2 - g0, g2*g2 + 1;
    R = R * factor;
    R.diagonal().array() -= 1.f;

    return R;
}

template<typename T>
Matrix3<T> gvec2mat(const Vector3<T>& gvec){
    return gvec2mat(&gvec[0]);
};

template<typename Derived>
Vector3<typename Derived::Scalar> mat2gvec(const Eigen::MatrixBase<Derived>& R){
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 3, "input must be 3x3");
    Vector3<typename Derived::Scalar> g;
    g << R(1, 2) - R(2, 1), R(2, 0) - R(0, 2), R(0, 1) - R(1, 0);
    g *= 1 / (R(0, 0) + R(1, 1) + R(2, 2) + 1);

    return g;
}

// b * a, i.e. apply a first, then apply b rotation, equivalent to gvec2mat(b) * gvec2mat(a)
template<typename Derived>
Vector3<typename Derived::Scalar> gvec_mul(const Eigen::MatrixBase<Derived>& b, const Eigen::MatrixBase<Derived>& a) {
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "input must be 3x1");
    return (a + b - b.cross(a)) / (1 - a.dot(b));
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
        orthogonalised(const Eigen::MatrixBase<Derived>& mat) {
#if 0 // via gvec, only works for 3x3
    return gvec2mat(mat2gvec(mat));
#else // by SVD
    auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    return svd.matrixU() * svd.matrixV().transpose();
#endif
};

template <typename Derived1, typename Derived2>
bool approx(const Eigen::MatrixBase<Derived1>& a, const Eigen::MatrixBase<Derived2>& b, const typename Derived1::Scalar threshold = 1E-4f){
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    return ((a - b).array().abs() < threshold * b.array().abs().cwiseMax(1)).all();
}

template <typename Derived1, typename Derived2>
bool approx(const Eigen::ArrayBase<Derived1>& a, const Eigen::ArrayBase<Derived2>& b, const typename Derived1::Scalar threshold = 1E-4f){
    assert(a.rows() == b.rows() && a.cols() == b.cols());
    return ((a - b).array().abs() < threshold * b.array().abs().cwiseMax(1)).all();
}

//find the minimum rotation from [0,0,1] to T.
//@todo: add unit test
//Tgvec[2] == 0.f
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> T2Tgvec(const Eigen::MatrixBase<Derived>& T){
    typedef typename Derived::Scalar scalar_type;
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "invalid T vector");
//    scalar_type xy_sqr_norm = T.template topRows<2>().squaredNorm();
//    scalar_type tan_half_theta_div_xy_norm = (T.norm() - T[2]) / xy_sqr_norm;
//T(2) == -T.norm() correspond to 180degree rotation 
    const scalar_type tan_half_theta_div_xy_norm = 1.f/(T.norm() + T[2]);
    Eigen::Matrix<scalar_type, 3, 1> Tgvec;
    Tgvec << T[1] * tan_half_theta_div_xy_norm, -T[0] * tan_half_theta_div_xy_norm, 0.f;
    return Tgvec;
}
//optimised version for T.norm() == 1
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1> UnitT2Tgvec(const Eigen::MatrixBase<Derived>& T){
    assert(std::abs(T.norm() - 1) < 1E-4f);
    typedef typename Derived::Scalar scalar_type;
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1, "invalid T vector");
    const scalar_type tan_half_theta_div_xy_norm = 1.f/(1 + T[2]);
    Eigen::Matrix<scalar_type, 3, 1> Tgvec;
    Tgvec << T[1] * tan_half_theta_div_xy_norm, -T[0] * tan_half_theta_div_xy_norm, 0.f;
    return Tgvec;
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3> T2TR(const Eigen::MatrixBase<Derived>& T){
    return gvec2mat(T2Tgvec(T));
}

template<typename T>
T sigmoid(T x)
{
    return std::tanh(x * 0.5) * 0.5 + 0.5;
}

template<typename T>
Matrix3<T> cross_vec2mat(const Vector3<T>& vec)
{
    Matrix3<T> mat;
    mat <<  0, -vec(2), vec(1),
            vec(2), 0, -vec(0),
            -vec(1), vec(0), 0;
    return mat;
}

template <typename T>
Vector3<T> cross_mat2vec(const Matrix3<T>& mat)
{
    Vector3<T> vec;
    vec <<  mat(2, 1) - mat(1, 2),
            mat(0, 2) - mat(2, 0),
            mat(1, 0) - mat(0, 1);
    return T(0.5f) * vec;
}

template <typename scalar>
Isometry3<scalar> make_isometry3(const Matrix3<scalar>& R, const Vector3<scalar>& T)
{
    Isometry3<scalar> RT(R);
    RT.translation() = T;
    return RT;
}

template<typename T>
std::vector<std::vector<uint32_t>> group_links(const std::vector<std::array<T, 2>> &links) {
//    std::set<T> vertices;
    typedef boost::typed_identity_property_map<T> identity_pmap_t;
    boost::disjoint_sets_with_storage<identity_pmap_t, identity_pmap_t> vertex_groups;

    for(const auto& link : links){
//        for(auto id : link)
//            vertices.emplace(id);
        vertex_groups.link(link[0], link[1]);
    }

    std::map<T, std::vector<uint32_t>> link_group_map;
    for(uint32_t i = 0; i < links.size(); i++){
        const T id0 = links[i][0];
        T parent = vertex_groups.find_set(id0);
        assert(vertex_groups.find_set(links[i][1]) == parent);
        link_group_map[parent].push_back(i);
    }

    std::vector<std::vector<uint32_t>> link_groups(link_group_map.size());
    std::transform(link_group_map.begin(), link_group_map.end(), link_groups.begin(), [](const std::pair<const T, std::vector<uint32_t>>& item){
        return std::move(item.second);
    });
    return link_groups;
}

template<typename Derived1, typename Derived2>
auto solve_GaussElim(const Eigen::MatrixBase<Derived1>& A, const Eigen::MatrixBase<Derived2>& b){
    typedef typename Derived1::Scalar ElemType;
    static_assert(std::is_same<typename Derived1::Scalar, typename Derived2::Scalar>::value, "fatal error");
    assert(A.rows() == b.rows() && A.rows() == A.cols());
    Eigen::Matrix<ElemType, A.RowsAtCompileTime, (A.ColsAtCompileTime < 0 || b.ColsAtCompileTime < 0) ? -1 :A.ColsAtCompileTime + b.ColsAtCompileTime> Ab(A.rows(), A.cols() + b.cols());
    for(int r = 0; r < A.rows(); r++){
        Ab.row(r) << A.row(r), b.row(r);
    }
    for (int i = 0; i < A.rows(); i++) {
        const ElemType inv = 1.f / Ab(i, i);
        for (int j = i + 1; j < A.rows(); j++) {
            const ElemType factor = Ab(j, i) * inv;
            for (int k = i + 1; k < A.cols() + b.cols(); k++) {
                Ab(j, k) -= factor * Ab(i, k);
            }
        }
    }
    Eigen::Matrix<ElemType, b.RowsAtCompileTime, b.ColsAtCompileTime> x = Ab.template rightCols<b.ColsAtCompileTime>();
    for (int i = A.rows() - 1; i >= 0; i--) {
        const ElemType inv = 1.f / Ab(i, i);
        x.row(i) *= inv;
        for (int j = i - 1; j >= 0; j--){
            const ElemType factor = Ab(j, i);
            x.row(j) -= factor * x.row(i);
        }
    }
    return x;
}