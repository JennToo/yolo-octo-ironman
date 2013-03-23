#include <NeuralNet.hpp>
#include <Examples.hpp>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>

int main(int argc, char** argv) {
    std::cout << "Starting...\n";

    std::srand((unsigned)std::time(0));

    std::vector<unsigned> top;
    top.push_back(1);
    top.push_back(3);
    top.push_back(3);
    top.push_back(1);
    ANN::NeuralNet network(top);

    std::vector<ANN::Example> examples;
    ANN::loadExamplesFromFile("solar_radiation.txt", examples, 1, 1);
    std::cout << "Loaded " << examples.size() << " examples from file\nOriginals:\n";
    ANN::printExamples(examples);

    ANN::Transformation t = ANN::getTransformation(examples);
    ANN::applyTransformation(examples, t, false);
    std::cout << "After transform:\n";
    ANN::printExamples(examples);

    std::cout << std::setprecision(6) << std::fixed;

    network.train(examples, 1.0, 0.0, 100000);
    for(std::size_t i = 0; i < examples.size(); i++) {
        network.computeActivation(examples[i].input);
        ANN::Output out = network.getActivation();
        double ex = examples[i].output.values[0];
        double got = out.values[0];
        /*std::cout << "Expected: " << ex
                  << " Got: " << got
                  << " Error%: " << 100 * std::fabs((ex - got) / ex)
                  << std::endl;*/
	std::cout << ex << "," << got << std::endl;
    }

    return 0;
}
