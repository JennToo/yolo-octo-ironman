#pragma once

#include <cmath>
#include <cstdlib>
#include <numeric>

namespace ANN {
/**
 * Computes the sigmoid function. Used in NeuralNet activation
 */
inline double sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

/**
 * Computes the derivative of the sigmoid function. Used in
 * NeuralNet training
 */
inline double sigmoid_prime(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

/**
 * Returns a random value within the give range
 */
inline double randRange(double low, double high)
{
    double condense = (double)(std::rand()) / (double)(RAND_MAX);
    return condense * (high - low) + low;
}

/**
 * Compares two floating point values to see if they are within a
 * tolerence of each other
 */
inline bool tol_equal(double val1, double val2, double tol = 0.00001)
{
    return std::fabs(val1 - val2) < tol;
}

/**
 * Clamps the sum of all items in the vector to (approximately)
 * 1.0
 */
inline void normalize_vector(std::vector<double>& vec)
{
    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    for(std::size_t i = 0; i < vec.size(); i++) {
        vec[i] = vec[i] / sum;
    }
}
}
