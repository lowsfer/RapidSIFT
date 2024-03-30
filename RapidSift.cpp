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

#include "RapidSift.h"
#include "utils_host.h"
#include <vector>
#include "fp16.h"
#include "sift_master.h"

std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>
sort_kpoints(const std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>>& kpoints){
    require(kpoints.first.size() == kpoints.second.size());
    std::vector<int> indices(kpoints.first.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b){
        auto const& x = kpoints.first[a];
        auto const& y = kpoints.first[b];
        if (x.response != y.response) {
            return half2float(kpoints.first[a].response) > half2float(kpoints.first[b].response);
        }
        if (x.size != y.size) {
            return half2float(x.size) > half2float(y.size);
        }
        if (x.location.y != y.location.y) {
            return x.location.y < y.location.y;
        }
        if (x.location.x != y.location.x) {
            return x.location.x < y.location.x;
        }
        return x.angle < y.angle;
    });
    std::pair<std::vector<KeyPoint>, std::vector<SiftDescriptor>> result;
    result.first.resize(kpoints.first.size());
    result.second.resize(kpoints.second.size());
    std::transform(indices.begin(), indices.end(), result.first.begin(), [&](int i){return kpoints.first[i];});
    std::transform(indices.begin(), indices.end(), result.second.begin(), [&](int i){return kpoints.second[i];});
    return result;
}

extern "C" RapidSift *create_sift(size_t num_works, DescType descType) {
    return new sift_master(num_works, descType);
}
