#pragma once

#include <vector>
#include <string>

#include "Neuron.hpp"
#include "Connection.hpp"

namespace ANN {
    struct Layer {
        std::vector<Neuron*> neurons;
    };

    struct Input {
        std::vector<double> values;
    };

    struct Output {
        std::vector<double> values;
    };

    struct Example {
        Input in;
        Output out;
    };

    class NeuralNet {
        std::vector<unsigned> topology;
        std::vector<Layer> layers;


        // Only used to delete the graph on destructor
        std::vector<Neuron*> graph;
    public:
        NeuralNet(const std::vector<unsigned>& topology);
        NeuralNet(const std::string& file);
        ~NeuralNet();

        void computeActivation(const Input& input);
        std::vector<double> getActivation() const;
        void train(const std::vector<Example>& examples, double alpha, double stop, unsigned maxIterations);
    };
}
