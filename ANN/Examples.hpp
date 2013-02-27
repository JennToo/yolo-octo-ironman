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

    struct Transformation {
        double offset;
        double span;
    };

    Transformation getTransformation(const std::vector<Example>& examples);
    void applyTransformation(Example& ex, const Transformation& trans, bool inverse = false);
    void applyTransformation(std::vector<Example>& examples, const Transformation& trans, bool inverse = false);

    void loadExamplesFromFile(const std::string& file, std::vector<Example>& output);
}
