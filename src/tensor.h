#pragma once
#include <vector>
#include <string>

struct Tensor {
    void* data;
    std::vector<int> shape;
    std::string dtype;  // "fp16", "bf16", "fp32"
    size_t size() const {
        size_t total = 1;
        for (int d : shape) total *= d;
        return total;
    }
};
