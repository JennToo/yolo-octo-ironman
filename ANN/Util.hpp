#pragma once

#include <cmath>
#include <cstdlib>

namespace ANN {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double randRange(double low, double high) {
	double condense = (double)(std::rand()) / (double)(RAND_MAX);
	return condense * (high - low) + low;
    }
}
