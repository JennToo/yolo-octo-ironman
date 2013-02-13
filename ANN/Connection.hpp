#pragma once

namespace ANN {
    class Neuron;

    class Connection {
    public:
        Neuron* send;
        Neuron* recv;
        double weight;

        Connection(Neuron*, Neuron*, double);
    };
}
