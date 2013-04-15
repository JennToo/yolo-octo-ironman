#pragma once

namespace ANN {
    class Ensemble {
        std::vector<NeuralNetwork*> learners;
        std::vector<double> learnerWeights;
        std::vector<DiscreteClassifier> classifiers;
    public:
        void adaBoost(const Examples& examples, unsigned k);

        void classify(const Input& input, Output& output);
        void classify(const Example& example, Output& output) {
            classify(example.inputs, output);
        }
    };
}
