#include "NeuralNet.hpp"

#include <iostream>
#include <cstdlib>

namespace ANN {
    NeuralNet::NeuralNet(const std::vector<unsigned>& topology) {
        this->topology = topology;

        // Generate each layer
        for(std::size_t lay = 0; lay < topology.size(); lay++) {
            unsigned layerSize = topology[lay];
            Layer layer;

            // Generate each neuron on current layer
            for(unsigned i = 0; i < layerSize; i++) {
                Neuron* neuron = new Neuron();

                // The input (first) layer doesn't have any inputs
                // to connect to
                if(lay != 0) {
                    Layer& prev = layers[lay-1];
                    for(std::size_t prevI = 0; prevI < prev.neurons.size(); prevI++) {
                        Connection* con = new Connection(prev.neurons[prevI], neuron, 0.0);
                    }

                    // Bias node isn't actually stored in a layer
                    Neuron* bias = new Neuron();
                    bias->setActivation(1.0);
                    graph.push_back(bias);
                    Connection* con = new Connection(bias, neuron, 0.0);
                }
                layer.neurons.push_back(neuron);
                graph.push_back(neuron);
            }
            layers.push_back(layer);
        }
    }

    NeuralNet::NeuralNet(const std::string& file) {
        std::cerr << "Loading from file is stub\n";
        std::exit(1);
    }

    NeuralNet::~NeuralNet() {
        for(std::size_t i = 0; i < graph.size(); i++) {
            for(std::size_t j = 0; j < graph[i]->inputs.size(); j++) {
                delete graph[i]->inputs[j];
            }
            delete graph[i];
        }
    }

    void NeuralNet::computeActivation(const Input& input) {
        // Set the input activations
        for(std::size_t i = 0; i < layers[0].neurons.size(); i++) {
            layers[0].neurons[i]->setActivation(input.values[i]);
        }
    }
    std::vector<double> NeuralNet::getActivation() const {

    }

    void NeuralNet::train(const std::vector<Example>& examples, double alpha,
                          double stop, unsigned maxIterations) {
        
    }
}
