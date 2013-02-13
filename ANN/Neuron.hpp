#pragma once

#include <vector>

#include "Connection.hpp"

namespace ANN {
    class Neuron {
        double activation;
        double input;
        double delta;
    public:
        // Breaking the rules in the name of speed
        std::vector<Connection*> inputs;
        std::vector<Connection*> outputs;

        Neuron() : activation(0), input(0), delta(0) {
        }

        double getActivation() const {
            return activation;
        }

        double getDelta() const {
            return delta;
        }

        void setActivation(double act) {
            activation = act;
        }

        void setDelta(double del) {
            delta = del;
        }
    };
}
