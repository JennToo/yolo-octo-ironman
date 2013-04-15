#pragma once

#include <vector>

#include "Examples.hpp"

namespace ANN {
    enum ClassifierMethod {
        FLOOR, CEILING, ROUND
    };

    class DiscreteClassifier {
        std::vector<double> values;
    public:
        DiscreteClassifier(const std::vector<double>& values) : values(values) {
        }

        double getIndexValue(std::size_t index) const;
        std::size_t getClassificationIndex(double continuousValue,
                                           ClassifierMethod method) const;
        static bool correctClassification(const std::vector<DiscreteClassifier>& cls,
                                          const Output& compOut, const Output& exOut);
    };
}
