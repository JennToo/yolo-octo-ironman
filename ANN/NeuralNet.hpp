#pragma once

#include <vector>
#include <string>

#include "Neuron.hpp"
#include "Connection.hpp"
#include "Examples.hpp"

namespace ANN {
    struct Layer {
        std::vector<Neuron*> neurons;
    };

    class NeuralNet {
        std::vector<unsigned> topology;
        std::vector<Layer> layers;


        // Only used to delete the graph on destructor
        std::vector<Neuron*> graph;

        void trainExample(const Example& example, double alpha);
    public:
        NeuralNet(const std::vector<unsigned>& topology);
        NeuralNet(const std::string& file);
        ~NeuralNet();

        void computeActivation(const Input& input);
        Output getActivation() const;
        double getTotalError(const Example& ref);
        void train(const std::vector<Example>& examples, double alpha, double stop, unsigned maxIterations);
    };
}
