#include "Ensemble.hpp"
#include <map>

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

    void Ensemble::classify(const Input& input, Output& output) {
        std::vector<std::vector<std::size_t>> votes;

        for(std::size_t i = 0; i < learners.size(); i++) {
            std::vector<std::size_t> values;
            // Compute the current learner's output
            NeuralNetwork* net = learners[i];
            net->computeActivation(input);
            Output compOut = net->getActivation();

            // Add each output to the vote (the classifier index, to
            // allow for == comparison later)
            for(std::size_t out = 0; out < compOut.values.size(); out++) {
                std::size_t index = clasifiers[out].getClassificationIndex(compOut.values[out],
                                                                           ClassifierMethod::ROUND);
                values.push_back(index);
            }
            votes.push_back(values);
        }
        output.values.clear();

        // Get votes for each output
        for(std::size_t out = 0; out < votes[0].size(); out++) {
            // Maps each choice to a vote weight
            std::map<std::size_t, double> tally;
            for(std::size_t i = 0; i < learners.size(); i++) {
                std::size_t vote = votes[i][out];
                tally[vote] += learnerWeights[i];
            }

            std::size_t maxVoteIndex = (*tally.begin()).first;
            double maxVoteWeight = (*tally.begin()).second;
            for(std::map<std::size_t, double>::iterator i = tally.begin();
                i != tally.end();
                i++) {
                if((*i).second > maxVoteWeight) {
                    maxVoteIndex = (*i).first;
                    maxVoteWeight = (*i).second;
                }
            }

            output.values.push_back(classifiers[out].getIndexValue(maxVoteIndex));
        }
    }
}
