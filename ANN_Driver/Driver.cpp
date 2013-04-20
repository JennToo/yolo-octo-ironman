#include <NeuralNet.hpp>
#include <Examples.hpp>
#include <Ensemble.hpp>
#include <Util.hpp>

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
    top.push_back(11);
    top.push_back(6);
    top.push_back(5);
    top.push_back(1);
    ANN::NeuralNet network(top);

    std::vector<ANN::Example> examples;
    ANN::loadExamplesFromFile("red_wine_quality.txt", examples, 11, 1);
    std::cerr << "Loaded " << examples.size() << " examples from file\nOriginals:\n";
    //ANN::printExamples(examples);

    ANN::Transformation t = ANN::getTransformation(examples);
    ANN::applyTransformation(examples, t, false);
    std::cout << "After transform:\n";
    //ANN::printExamples(examples);

    //std::cout << std::setprecision(6) << std::fixed;

    ANN::Ensemble learner;
    learner.adaBoost(examples, 300);
    int incorrect = 0;
    for(std::size_t i = 0; i < examples.size(); i++) {
	ANN::Output classification;
	learner.classify(examples[i].input, classification);
	double ex = examples[i].output.values[0];
        double got = classification.values[0];
        std::cout << "Expected: " << ex
                  << " Got: " << got
                  << std::endl;
	if(!ANN::tol_equal(ex, got))
	    incorrect++;
    }

    std::cout << "Misclassified " << incorrect << " examples ("
	      << 100 * std::fabs((double)incorrect / examples.size())
	      << "%)\n";

    /*network.train(examples, 0.1, 0.0, 10000);
    for(std::size_t i = 0; i < examples.size(); i++) {
        network.computeActivation(examples[i].input);
        ANN::Output out = network.getActivation();
        double ex = examples[i].output.values[0];
        double got = out.values[0];
        std::cout << "Expected: " << ex
                  << " Got: " << got
                  << " Error%: " << 100 * std::fabs((ex - got) / ex)
                  << std::endl;
	std::cout << ex << "," << got << std::endl;
    }*/

    return 0;
}
