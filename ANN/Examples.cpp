#include "Examples.hpp"

#include <limits>

namespace ANN {
    Transformation getTransformation(const std::vector<Example>& examples) {
        Transformation toRet;

        // find max and min
        double min = std::numeric_limits<double>::infinity();
        double max = -min;
        //TODO loop and find

        toRet.offset = -min;
        toRet.span = (max - min);

        return toRet;
    }

    void applyTransformation(Example& ex, const Transformation& trans, bool inverse) {
        //TODO
    }

    void applyTransformation(std::vector<Example>& examples, const Transformation& trans, bool inverse) {
        //TODO
    }

    void loadExamplesFromFile(const std::string& file, std::vector<Example>& output) {
        //TODO
    }

}
