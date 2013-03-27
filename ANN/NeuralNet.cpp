#include "NeuralNet.hpp"
#include "Util.hpp"

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <algorithm>

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
                    Layer* prev = &layers[lay-1];

                    for(std::size_t prevI = 0; prevI < prev->neurons.size(); prevI++) {
                        new Connection(prev->neurons[prevI], neuron, randRange(-0.1, 0.1));
		    }

                    // Bias node isn't actually stored in a layer
                    Neuron* bias = new Neuron();
                    bias->setActivation(1.0);
                    graph.push_back(bias);
                    new Connection(bias, neuron, randRange(-0.1, 0.1));
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

    void NeuralNet::computeActivation(const Input& inputVals) {
        // Set the input activations
        for(std::size_t i = 0; i < layers[0].neurons.size(); i++) {
            layers[0].neurons[i]->setActivation(inputVals.values[i]);
        }

        // Set activation of each layer in sequence
        for(std::size_t layer = 1; layer < layers.size(); layer++) {
            Layer& lay = layers[layer];
            // Set activation of each node in current layer
            for(std::size_t j = 0; j < lay.neurons.size(); j++) {
                Neuron* neuron = lay.neurons[j];
                double input = 0.0;

                // Sum activation * weight of each connected Neuron
                // from previous layer
                for(std::size_t i = 0; i < neuron->inputs.size(); i++) {
                    Connection* con = neuron->inputs[i];
                    assert(con->recv == neuron);
                    input += con->weight * con->send->getActivation();
                }

                neuron->setInput(input);
                neuron->setActivation(sigmoid(input));
            }
        }
    }

    Output NeuralNet::getActivation() const {
        Output toRet;

        // The last layer is the output layer
        const Layer& layer = layers[layers.size()-1];
        for(std::size_t i = 0; i < layer.neurons.size(); i++) {
            toRet.values.push_back(layer.neurons[i]->getActivation());
        }

        return toRet;
    }

    double NeuralNet::getTotalError(const Example& ref) {
        // Get net's current best classification
        computeActivation(ref.input);
        Output netout = getActivation();

        double error = 0.0;
        for(std::size_t i = 0; i < netout.values.size(); i++) {
            error += fabs(netout.values[i] - ref.output.values[i]);
        }

        return error;
    }

    void NeuralNet::printNetwork() const {
        for(std::size_t i = 0; i < layers.size(); i++) {
            for(std::size_t n = 0; n < layers[i].neurons.size(); n++) {
                Neuron* neuron = layers[i].neurons[n];
                std::cout << "Neuron " << i << ":" << n << std::endl;
                std::cout << "   Act: " << neuron->getActivation() << " Delta: " << neuron->getDelta() << std::endl;
                std::cout << "   Input count: " << neuron->inputs.size() << " Output count: " << neuron->outputs.size() << std::endl;
                std::cout << "   Input Weights:\n";
                for(std::size_t c = 0; c < neuron->inputs.size(); c++) {
                    std::cout << "      " << neuron->inputs[c]->weight << " <- " << i+1 << ":" << c
                              << " (Input activation: " << neuron->inputs[c]->send->getActivation() << ")" << std::endl;
                }
                std::cout << "   Output Weights:\n";
                for(std::size_t c = 0; c < neuron->outputs.size(); c++) {
                    std::cout << "      " << neuron->outputs[c]->weight << " -> " << i+1 << ":" << c << std::endl;
                }
                std::cout << std::endl;
            }
        }
    }

    void NeuralNet::train(const std::vector<Example>& examples, double alpha,
                          double stop, unsigned maxIterations) {
        double error;
        unsigned iterations = 0;
        std::vector<Example> shuffled(examples);

        do {
            // Shuffling the examples speeds up process greatly
            std::random_shuffle(shuffled.begin(), shuffled.end());

            // Process each example
            std::vector<Example>::iterator iter;
            for(iter = shuffled.begin(); iter != shuffled.end(); iter++) {
                trainExample(*iter, alpha);
                error += getTotalError(*iter);
            }

            error = error / examples.size();
            iterations++;

	    if(iterations % 1000 == 0) {
		std::cout << "After " << iterations << " iteraitons error is: " << error << std::endl;
	    }
        } while(error > stop && iterations <= maxIterations);
    }

    void NeuralNet::trainExample(const Example& example, double alpha) {
        computeActivation(example.input);

        // Compute initial deltas on output layer
        Layer& outputLayer = layers[layers.size() - 1];
        for(std::size_t i = 0; i < outputLayer.neurons.size(); i++) {
            Neuron* neuron = outputLayer.neurons[i];
            double activ = neuron->getActivation();

            // s'(x) = s(x) * (1 - s(x)) for optimization
            // preventing two calls to exp(x)
            double deriv = activ * (1 - activ);

            neuron->setDelta(deriv * (example.output.values[i] - activ));
        }

        // Operate on each layer backwards
        for(std::size_t layerI = layers.size()-2; layerI > 0; layerI--) {
            Layer& layer = layers[layerI];

            // Compute delta for each node in layer
            for(std::size_t i = 0; i < layer.neurons.size(); i++) {
                Neuron* neuron = layer.neurons[i];
                double activ = neuron->getActivation();
                double deriv = activ * (1 - activ);

                // Sum over next layer's deltas
                double sum = 0.0;
                for(std::size_t j = 0; j < neuron->outputs.size(); j++) {
                    Connection* con = neuron->outputs[j];

                    sum += con->weight * con->recv->getDelta();
                }

                neuron->setDelta(sum * deriv);
            }
        }

        // Apply delta to weights
        for(std::size_t layerI = 0; layerI < layers.size(); layerI++) {
            Layer& layer = layers[layerI];

            for(std::size_t j = 0; j < layer.neurons.size(); j++) {
                Neuron* neuron = layer.neurons[j];

                // Update each input connection
                for(std::size_t i = 0; i < neuron->inputs.size(); i++) {
                    Connection* con = neuron->inputs[i];
                    con->weight = con->weight + alpha * con->send->getActivation() * neuron->getDelta();
                }
            }
        }
    }
}
