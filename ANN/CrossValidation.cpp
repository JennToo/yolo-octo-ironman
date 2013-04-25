#include "CrossValidation.hpp"
#include "NeuralNet.hpp"

#include <algorithm>
#include <cmath>

namespace ANN {
typedef std::vector<Example> Examples;

// Break examples into sets
void partition(std::vector<Examples>& sets,
               const std::vector<Example>& examples, unsigned bins)
{
    const unsigned perBin = examples.size() / bins;
    for(unsigned i = 0; i < bins; i++) {
        const unsigned offset = perBin * i;
        sets.push_back(std::vector<Example>(examples.begin() + offset, examples.begin() + offset + perBin));
    }

    // Grab any leftovers
    std::size_t bin = 0;
    for(unsigned i = perBin * bins; i < examples.size(); i++) {
        sets[bin].push_back(examples[i]);
    }
}

// Recombines all examples except one into a single set
void reconcile(const std::vector<Examples>& sets,
               std::vector<Example>& store, unsigned leaveOutBin)
{
    for(std::size_t i = 0; i < sets.size(); i++) {
        if(i != leaveOutBin) {
            store.insert(store.end(), sets[i].begin(), sets[i].end());
        }
    }
}

// Performs one round of validation on the given bin, after training
double checkBin(unsigned bin, const std::vector<Examples>& sets,
                const std::vector<unsigned>& topology, unsigned total)
{
    NeuralNet network(topology);
    std::vector<Example> toTrainOn;
    reconcile(sets, toTrainOn, bin);

    // Train on training set
    Transformation t = getTransformation(toTrainOn);
    applyTransformation(toTrainOn, t, false);
    network.train(toTrainOn, 1.0, 0.001, 4000 * total);

    // Compute average error in validation set, unscaled
    const std::vector<Example>& validation = sets[bin];
    double error;
    for(std::size_t i = 0; i < validation.size(); i++) {
        const Example& original = validation[i];
        network.computeActivation(original.input);
        Output netOut = network.getActivation();
        applyTransformation(netOut, t, true);

        for(std::size_t j = 0; j < netOut.values.size(); j++) {
            error += std::fabs(netOut.values[j] - original.output.values[j]);
        }
    }
    return error / validation.size();
}

double crossValidation(const std::vector<Example>& examples,
                       const std::vector<unsigned>& topology, unsigned bins)
{
    std::vector<Example> shuff(examples);
    std::random_shuffle(shuff.begin(), shuff.end());

    std::vector<Examples> sets;
    partition(sets, shuff, bins);

    double error = 0.0;
    for(unsigned i = 0; i < bins; i++) {
        error += checkBin(i, sets, topology, examples.size());
    }

    return error / bins;
}
}
