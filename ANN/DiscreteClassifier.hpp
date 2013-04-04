#pragma once

#include <vector>

namespace ANN {
    enum ClassifierMethod {
        FLOOR, CEILING, ROUND
    };

    class DiscreteClassifier {
        std::vector<double> values;
    public:
        DiscreteClassifier(const std::vector<double>& values) : values(values) {
        }

        double getIndexValue(std::size_t index);
        std::size_t getClassificationIndex(double continuousValue,
                                           ClassifierMethod method);
    };
}
