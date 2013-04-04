#pragma once

#include <cmath>
#include <cstdlib>

namespace ANN {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_prime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    double randRange(double low, double high) {
	double condense = (double)(std::rand()) / (double)(RAND_MAX);
	return condense * (high - low) + low;
    }

    bool tol_equal(double val1, double val2, double tol = 0.00001) {
        return std::fabs(val1 - val2) < tol;
    }
}
