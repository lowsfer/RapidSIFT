//
// Created by yao on 15/01/18.
//

#pragma once
#include <cuda_runtime_api.h>

//@todo: check correctness!
template <typename T>
__host__ __device__ __forceinline__ void solve_GaussElim(const T(&A)[3][3], const T(&b)[3], T(&x)[3]){
    T Ab[3][4] = {
            A[0][0], A[0][1], A[0][2], b[0],
            A[1][0], A[1][1], A[1][2], b[1],
            A[2][0], A[2][1], A[2][2], b[2]
    };
    for (int i = 0; i < 3; i++) {
        const T inv = 1.f / Ab[i][i];
        for (int j = i + 1; j < 3; j++) {
            const T factor = Ab[j][i] * inv;
            for (int k = 0; k < 4; k++) {
                if(k > i)
                    Ab[j][k] -= factor * Ab[i][k];
            }
        }
    }
    for (int i = 2; i >= 0; i--) {
        const T inv = 1.f / Ab[i][i];
        Ab[i][3] *= inv;
        x[i] = Ab[i][3];
        for (int j = i - 1; j >= 0; j--){
            const T factor = Ab[j][i];
            Ab[j][3] -= factor * Ab[i][3];
        }
    }
}

template <typename T>
__host__ __device__ __forceinline__ void decompose_LU(const T(&M)[3][3], T(&LU)[3][3]){
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
            LU[i][j] = M[i][j];
    for (int i = 0; i < 3; i++) {
        const T inv = 1.f / LU[i][i];
        for (int j = i + 1; j < 3; j++) {
            const T factor = LU[j][i] * inv;
            LU[j][i] = factor;
            for (int k = 0; k < 3; k++) {
                if(k > i)
                    LU[j][k] -= factor * LU[i][k];
            }
        }
    }
}