#include "Connection.hpp"
#include "Neuron.hpp"

namespace ANN {
    Connection::Connection(Neuron* send, Neuron* recv, double weight)
        : send(send), recv(recv), weight(weight) {
        // Register the connection with each node
        send->outputs.push_back(this);
        recv->inputs.push_back(this);
    }
}
