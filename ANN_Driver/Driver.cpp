#include <NeuralNet.hpp>

#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Starting...\n";
    std::vector<unsigned> top;
    top.push_back(1);
    top.push_back(3);
    top.push_back(3);
    top.push_back(1);
    ANN::NeuralNet network(top);

    std::vector<double> input;
    input.push_back(0.314);
    ANN::Input in;
    in.values = input;
    network.computeActivation(in);

    ANN::Output out = network.getActivation();
    std::cout << out.values[0] << std::endl;

    return 0;
}
