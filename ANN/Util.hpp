#pragma once

#include <cmath>

namespace ANN {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
}
