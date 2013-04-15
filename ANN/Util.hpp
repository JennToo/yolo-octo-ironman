#pragma once

#include <cmath>
#include <cstdlib>
#include <algorithm>

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

    void normalize_vector(std::vector<double>& vec) {
        double max = std::max_element(vec.begin(), vec.end());
        for(std::size_t i = 0; i < vec.size(); i++) {
            vec[i] /= max;
        }
    }
}
