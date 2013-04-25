#pragma once

#include <vector>
#include <string>

namespace ANN {
struct Input {
    std::vector<double> values;
};

struct Output {
    std::vector<double> values;
};

struct Example {
    Input input;
    Output output;
};

typedef std::vector<Example> Examples;

struct Transformation {
    std::vector<double> offset;
    std::vector<double> span;
};

Transformation getTransformation(const std::vector<Example>& examples);
void applyTransformation(Output& output, const Transformation& trans, bool inverse);
void applyTransformation(Example& ex, const Transformation& trans, bool inverse = false);
void applyTransformation(std::vector<Example>& examples, const Transformation& trans, bool inverse = false);

void loadExamplesFromFile(const std::string& file, std::vector<Example>& output, std::size_t inputs, std::size_t outputs);
void printExamples(const std::vector<Example>& examples);
}
