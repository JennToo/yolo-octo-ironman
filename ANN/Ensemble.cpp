#include "Ensemble.hpp"
#include "Util.hpp"
#include <map>
#include <cmath>
#include <iostream>

namespace ANN {
    bool correctClassification(const std::vector<DiscreteClassifier>& classifiers,
                                      const Output& compOut, const Output& exOut) {
        for(std::size_t out = 0; out < classifiers.size(); out++) {
            const DiscreteClassifier& classifier = classifiers[out];
            std::size_t i = classifier.getClassificationIndex(compOut.values[out], ClassifierMethod::ROUND);
            bool correct = tol_equal(classifier.getIndexValue(i), exOut.values[out]);
            if(!correct)
                return false;
        }
        return true;
    }

    void Ensemble::adaBoost(const Examples& examples, unsigned k) {
        std::vector<double> exampleWeights;
        double initial = 1.0 / examples.size();
        exampleWeights.resize(examples.size(), initial);

	// Create classifiers
	for(std::size_t i = 0; i < examples[0].output.values.size(); i++) {
	    std::vector<double> values;
	    for(std::size_t j = 0; j < examples.size(); j++) {
		values.push_back(examples[j].output.values[i]);
	    }

	    classifiers.push_back(DiscreteClassifier(values));
	}

        // Single topology to start

	bool kept = true;
        // Generate learners
        for(std::size_t i = 0; i < k; i++) {
	    if(!kept) {
		kept = true;
		i--;
	    }

	    std::cout << "Network #" << i << std::endl;

	    std::vector<unsigned> topology = {11, std::rand() % 5 + 4, std::rand() % 6 + 6, std::rand() % 6 + 6, 1};

            NeuralNet* network = new NeuralNet(topology);
            network->weightedTrain(examples, exampleWeights, 0.1, 0.0, 600000);

            // Check the classifications
            std::vector<bool> classifications;

            for(std::size_t j = 0; j < examples.size(); j++) {
                const Example& example = examples[j];
                network->computeActivation(example.input);
                Output activation = network->getActivation();

                classifications.push_back(correctClassification(classifiers, activation, example.output));
            }

            // Find out how poor the classifier is
            double error = 0.0;
            for(std::size_t j = 0; j < examples.size(); j++) {
                if(!classifications[j])
                    error += exampleWeights[j];
            }

	    if(error > 0.55 && i > 0) {
		std::cout << "Ommitting error: " << error << std::endl;
		kept = false;
		continue;
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
	    std::cout << "Weighted_Error " << error << std::endl;
            learners.push_back(network);
            learnerWeights.push_back(std::log((1-error)/error));


	    // Unweighted training error (for debug)
	    int incorrect = 0;
	    for(std::size_t ii = 0; ii < examples.size(); ii++) {
		Output classification;
		classify(examples[ii].input, classification);
		double ex = examples[ii].output.values[0];
		double got = classification.values[0];
		if(!tol_equal(ex, got))
		    incorrect++;
	    }
	    std::cout << "Unweighted_Error " << (float)(incorrect) / examples.size() << std::endl;

        }

	std::cout << "Final weights for each network: \n";
	for(std::size_t i = 0; i < learnerWeights.size(); i++) {
	    std::cout << learnerWeights[i] << std::endl;
	}
    }

    void Ensemble::classify(const Input& input, Output& output) {
        std::vector<std::vector<std::size_t>> votes;

        for(std::size_t i = 0; i < learners.size(); i++) {
            std::vector<std::size_t> values;
            // Compute the current learner's output
            NeuralNet* net = learners[i];
            net->computeActivation(input);
            Output compOut = net->getActivation();

            // Add each output to the vote (the classifier index, to
            // allow for == comparison later)
            for(std::size_t out = 0; out < compOut.values.size(); out++) {
                std::size_t index = classifiers[out].getClassificationIndex(compOut.values[out],
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
