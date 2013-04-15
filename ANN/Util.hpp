#pragma once

#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace ANN {
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoid_prime(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    inline double randRange(double low, double high) {
	double condense = (double)(std::rand()) / (double)(RAND_MAX);
	return condense * (high - low) + low;
    }

    inline bool tol_equal(double val1, double val2, double tol = 0.00001) {
        return std::fabs(val1 - val2) < tol;
    }

    inline void normalize_vector(std::vector<double>& vec) {
        double max = *(std::max_element(vec.begin(), vec.end()));
        for(std::size_t i = 0; i < vec.size(); i++) {
            vec[i] /= max;
        }
    }
}
