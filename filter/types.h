#pragma once

#include <eigen3/Eigen/Dense>
#include <boost/functional/hash_fwd.hpp>
#include <opencv2/core.hpp>
#include <array>
#include <boost/optional.hpp>
#include <cassert>
//#include "config.h"
#include <boost/format.hpp>
#include <iosfwd>
#include <vector>
#include <eigen3/Eigen/StdVector>
//using boost::optional;
using boost::format;

//typedef float fptype;

template<typename T>
using Isometry3 = Eigen::Transform<T, 3, Eigen::Isometry>;

template<typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template<typename T>
using Vector2 = Eigen::Matrix<T, 2, 1>;

template<typename T>
using Matrix3 = Eigen::Matrix<T, 3, 3>;

template<typename T>
using affine2_t = Eigen::Transform<T, 2, Eigen::AffineCompact>;

template<typename T>
using isometry2_t = Eigen::Transform<T, 2, Eigen::Isometry>;

typedef uint32_t img_id_t;
typedef uint32_t img_pair_id_t;

typedef uint32_t feature_id_t;

typedef uint32_t point_id_t;

struct observ_id_t{
    img_id_t image_id;
    feature_id_t feature_id;

    bool operator==(const observ_id_t& other) const {
        return image_id == other.image_id && feature_id == other.feature_id;
    }

    bool operator<(const observ_id_t& other) const {
        return image_id < other.image_id || (image_id == other.image_id && feature_id < other.feature_id);
    }
};

namespace std
{
    template<>
    struct hash<observ_id_t>
    {
        typedef observ_id_t argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& ob) const
        {
            std::size_t seed = 0;
            boost::hash_combine(seed, ob.image_id);
            boost::hash_combine(seed, ob.feature_id);
            return seed;
        }
    };

    template<typename T, size_t dim>
    struct hash<std::array<T, dim>>
    {
        typedef std::array<T, dim> argument_type;
        typedef std::size_t result_type;
        result_type operator()(argument_type const& arg) const{
            return boost::hash_range(arg.begin(), arg.end());
        }
    };
}


template<typename T>
struct vec2{
    T x, y;

    typename Vector2<T>::MapType map(){
        return Vector2<T>::Map(&x);
    }
    typename Vector2<T>::ConstMapType map() const {
        return Vector2<T>::Map(&x);
    }

    template <typename DstType>
    vec2<DstType> cast() const{
        return vec2<DstType>{
                DstType(x), DstType(y)
        };
    }
};

template<typename T>
struct vec3{
    T x, y, z;

    typename Vector3<T>::MapType map(){
        return Vector3<T>::Map(&x);
    }
    typename Vector3<T>::ConstMapType map() const {
        return Vector3<T>::Map(&x);
    }

    template <typename DstType>
    vec3<DstType> cast() const{
        return vec3<DstType>{
                DstType(x), DstType(y), DstType(z)
        };
    }
};





enum class feature_type_enum{
    akaze, orb
};

class scope_guard{
public:
    scope_guard(std::function<void()> func)
            :m_func(std::move(func)){}
    ~scope_guard(){
        if(m_func)
            m_func();
    }
private:
    std::function<void()> m_func;
};

template<typename T = float>
class camera_t
{
    typedef Eigen::Array<T, 2, 1> Array2;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    camera_t(){}

    camera_t(const Array2& f, const Array2& c)
            :f(f), c(c) {}

    camera_t(vec2<T> f, vec2<T> c)
            :f(f.map()), c(c.map()){}

    template<typename dst_type>
    camera_t<dst_type> cast(){
        return camera_t<dst_type>(f.template cast<dst_type>(), c.template cast<dst_type>());
    }

    Vector2<T> project(const Vector3<T>& pt) const{
        return f * pt.hnormalized().array() + c;
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 2> project_all(const Eigen::MatrixBase<Derived>& pts) const
    {
        return (pts.array().rowwise().hnormalized().rowwise() * f.transpose().template cast<typename Derived::Scalar>()).rowwise() + c.transpose().template cast<typename Derived::Scalar>();
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, 2> project_all(const Isometry3<typename Derived::Scalar>& RT, const Eigen::MatrixBase<Derived>& pts) const
    {
        typedef typename Derived::Scalar scalar_type;
        const Eigen::Matrix<scalar_type, 3, 4> P = K().template cast<scalar_type>() * RT.matrix.template topRows<3>();
        return (pts * P.transpose()).rowwise().hnormalized();
    }

    T fx() const{ return f[0]; }
    T fy() const{ return f[1]; }

    T cx() const {return c[0]; }
    T cy() const {return c[1]; }

    Eigen::Matrix<T, 3, 3> K() const {
        Eigen::Matrix<T, 3, 3> result;
        result <<
               fx(), 0, cx(),
                0, fy(), cy(),
                0,  0,  1;
        return result;
    }

    Array2 f;
    Array2 c;
};

template<typename T>
using eigen_aligned_vector = std::vector<T, Eigen::aligned_allocator<T>>;