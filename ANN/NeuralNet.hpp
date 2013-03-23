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

    /**
     * A fully connected, feed forward neural network implementation
     */
    class NeuralNet {
        std::vector<unsigned> topology;
        std::vector<Layer> layers;


        // Only used to delete the graph on destructor
        std::vector<Neuron*> graph;

        void trainExample(const Example& example, double alpha);
    public:
        /**
         * Constructs a neural network with the given topology giving
         * the number of neurons on each layer
         *
         * @param topology Vector giving the neuron count on each
         * layer. Must have at least two elements
         */
        NeuralNet(const std::vector<unsigned>& topology);

        /**
         * Constructs a neural network from a file. Currently STUB
         *
         * @param file The path to the file to load
         */
        NeuralNet(const std::string& file);

        ~NeuralNet();

        /**
         * Activates the neural network based on some input
         *
         * @param input Input values. Must be the same size as given
         * in the first value in the topology
         */
        void computeActivation(const Input& input);

        /**
         * Gathers the values of all output nodes
         *
         * @return The output values. Will be the same size as last
         * value in the topology
         */
        Output getActivation() const;

        /**
         * Computes the total error the network makes based on it's
         * current training.
         *
         * @param ref The example to check against. The output must
         * already be scaled down to [0,1]
         *
         * @return The sum total error for all outputs against the
         * example
         */
        double getTotalError(const Example& ref);

        /**
         * Trains the network using the passed example set. Uses basic
         * backpropogation.
         *
         * @param examples A set of examples to train against. The
         * outputs are expected to already be scaled to [0,1]
         *
         * @param alpha The learning rate to use during training. See
         * the backpropogation algorithm for details
         *
         * @param stop The average error for each example to consider
         * completion at
         *
         * @param maxIterations The maximum number of times to iterate
         * over the entire example set
         */
        void train(const std::vector<Example>& examples, double alpha, double stop, unsigned maxIterations);
    };
}
