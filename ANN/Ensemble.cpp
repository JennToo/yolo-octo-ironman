#include "Ensemble.hpp"

namespace ANN {
    void Ensemble::adaBoost(const Examples& examples, unsigned k) {
        std::vector<double> exampleWeights;
        double initial = 1.0 / examples.size();
        examplesWeights.resize(examples.size, initial);

        // Single topology to start
        std::vector<unsigned> topology = {11, 4, 1};

        // Generate learners
        for(std::size_t i = 0; i < k; i++) {
            NeuralNetwork* network = new NeuralNetwork(topology);
            network->weightedTrain(examples, exampleWeights, 0.1, 0.01, 1000);

            // Check the classifications
            std::vector<bool> classifications;

            for(std::size_t j = 0; j < examples.size(); j++) {
                const Example& example = examples[j];
                network->computeActivation(example.inputs);
                Output activation = network->getActivation();

                classifications.push_back(DiscreteClassifier::correctClassification(classifiers, activation, example.output));
            }

            // Find out how poor the classifier is
            double error = 0.0;
            for(std::size_t j = 0; j < examples.size(); j++) {
                if(!classifications[j])
                    error += exampleWeights[j];
            }

            // Adjust weights based on the classifier's error
            double factor = error / (1 - error);
            for(std::size_t j = 0; j < examples.size(); j++) {
                // Reduce weight of examples we got right
                if(classifications[j])
                    exampleWeights[j] *= factor;
            }

            normalize_vector(exampleWeights);

            // Store classifier with it's vote-weight
            learners.push_back(network);
            learnerWeights.push_back(std::log((1-error)/error));
        }
    }
}
