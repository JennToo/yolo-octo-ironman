#pragma once

#include "NeuralNet.hpp"
#include "DiscreteClassifier.hpp"
#include "Examples.hpp"

namespace ANN {
    class Ensemble {
        std::vector<NeuralNet*> learners;
        std::vector<double> learnerWeights;
        std::vector<DiscreteClassifier> classifiers;
    public:
	/**
	 * Implements the AdaBoost algorithm, storing the learners
	 * internally
	 *
	 * @param examples Examples to train on. Should be normalized
	 * already
	 *
	 * @param k Number of learners to create
	 */
        void adaBoost(const Examples& examples, unsigned k);

	/**
	 * Uses a weighted majority voting system to determine the
	 * classification of the given input
	 *
	 * @param input The input to compute on
	 *
	 * @param output The output vector to return in (should be
	 * empty to start)
	 */
        void classify(const Input& input, Output& output);

	/**
	 * Helper method that calls other classify method, extracting
	 * the input from example
	 */
        void classify(const Example& example, Output& output) {
            classify(example.input, output);
        }
    };
}
