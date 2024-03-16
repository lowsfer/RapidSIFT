cmake_minimum_required(VERSION 3.8)
project(RapidSIFT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75-real 86-real)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/../cuda-utils ${CMAKE_SOURCE_DIR}/../sosnet-ptq)
link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib64
        ${CMAKE_SOURCE_DIR}/../sosnet-ptq/build/vscode/
        /opt/TensorRT/lib
        ${CMAKE_SOURCE_DIR}/../cuda-utils/build/qtcreator-release/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wreturn-type -Wall -Wno-unknown-pragmas -Wextra -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(Eigen3 REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -keep -res-usage -Xptxas -v -Xptxas -warn-spills --expt-relaxed-constexpr -Wno-deprecated-gpu-targets --use_fast_math")

#find_library( NVTX_LIBRARY nvToolsExt PATHS ENV LD_LIBRARY_PATH /opt/cuda/lib64)

set(SOURCE_FILES
        RapidSift.h RapidSift.cpp
        conv1d.h types.h
        utils_sift.h
        kernel_DoG.cu
        kernel_DoG_HMMA.cu
        kernels.h
        kernels.cpp
        TexObj.h
        Pitched2DTexture.h
        utils_host.h
        utils.cpp
        Octave.h Octave.cpp
        SIFT_worker.h SIFT_worker.cpp
        cuda_async_copy_engine.h
        kernel_find_keypoints.cu
        kernel_assign_orientation.cu
        kernel_describe.cu
        kernel_makePatch.cu
        solve_GaussElim.h
        fp16.h
        public_types.h
        sift_master.h
        sift_master.cpp
        ThreadPool.h
        AdaptiveNMS.cu
        KArray.h AdaptiveNMS.h AdaptiveNMS.cpp)

#cuda_add_executable(texture_test texture_test.cu)

add_executable(tests
        tests/test_main.cpp
        tests/DoG_test.cpp
        tests/solve_GaussElim_test.cpp
        tests/find_keypoints_test.cpp
        tests/AdaptiveNMSTest.cpp)
target_compile_definitions(tests PRIVATE BOOST_TEST_DYN_LINK)
target_link_libraries(tests
#        ${CUDA_LIBRARIES}
        boost_unit_test_framework
        RapidSIFT
        opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_highgui
        )

add_library(RapidSIFT ${SOURCE_FILES})
target_link_libraries(RapidSIFT PRIVATE
        SOSNet_runtime sosnet_rc
        nvinfer
        cuda-utils)

find_package(OpenCV)
add_executable(conv1dtest main.cpp RapidSift.h filter/robust_gms_matcher.cpp filter/robust_gms_matcher.h
        filter/geometry.cpp filter/PropagationMatchFilter.cpp)
target_link_libraries(conv1dtest
        RapidSIFT
        opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs opencv_features2d
        opencv_calib3d
        nvToolsExt
        )

