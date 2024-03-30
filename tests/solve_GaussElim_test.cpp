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
// Created by yao on 15/01/18.
//

#include <boost/test/unit_test.hpp>
#include <random>
#include "../conv1d.h"
#include "../utils_sift.h"
#include "../types.h"
#include <opencv2/opencv.hpp>
#include "../kernels.h"
#include "../utils_host.h"
#include "../solve_GaussElim.h"
#include <cuda_runtime.h>
#include <eigen3/Eigen/Dense>

BOOST_AUTO_TEST_CASE(solve_GaussElim_test)
{
    float A[3][3];
    float b[3];
    float X[3];
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::MapType A_ref(&A[0][0]);
    A_ref.setRandom();
    Eigen::Vector3f::MapType b_ref(&b[0]);
    b_ref.setRandom();
    const Eigen::Vector3f X_ref = A_ref.fullPivLu().solve(b_ref);
    solve_GaussElim(A, b, X);
    const Eigen::Vector3f::MapType X_map(&X[0]);

    BOOST_CHECK_SMALL((X_ref - X_map).norm(), 1E-4f);
}

BOOST_AUTO_TEST_CASE(decompose_LU_test)
{
    float M[3][3];
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::MapType M_ref(&M[0][0]);
    M_ref.setRandom();
    Eigen::Matrix3f LU_ref = M_ref.partialPivLu().matrixLU();

    float LU[3][3];
    decompose_LU(M, LU);
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::MapType LU_map(&LU[0][0]);

    std::cout << "M = \n" << M_ref << std::endl;

    std::cout << "LU_ref = \n" << LU_ref << std::endl;
    std::cout << "LU = \n" << LU_map << std::endl;
}