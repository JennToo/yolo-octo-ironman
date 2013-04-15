#include "DiscreteClassifier.hpp"

namespace ANN {
    std::size_t DiscreteClassifier::getClassificationIndex(double continuousValue,
                                                          ClassifierMethod method) {
        std::size_t low = 0, high = values.size() - 1;
        while((high - low) > 1) {
            std::size_t mid = (high + low) / 2;
            if(continuousValue > values[mid]) {
                low = mid;
            } else {
                high = mid;
            }
        }

        switch(method) {
        default:
            return high;
        case ClassifierMethod::FLOOR:
            return low;
        case ClassifierMethod::CEILING:
            return high;
        case ClassifierMethod::ROUND:
            double distL = continuousValue - values[low];
            double distH = values[high] - continuousValue;
            return ((distH < distL) ? high : low);
        }
    }

    double DiscreteClassifier::getIndexValue(std::size_t index) {
        return values[index];
    }

    static bool correctClassification(const std::vector<DiscreteClassifier>& cls,
                                      const Output& compOut, const Output& exOut) {
        for(std::size_t out = 0; out < cls.size(); out++) {
            const DiscreteClassifier& classifer = classifiers[out];
            std::size_t i = classifier.getClassificationIndex(compOut.values[out]);
            bool correct = tol_equal(classifer.getIndexValue(i), exOut.values[out]);
            if(!correct)
                return false;
        }
        return true;
    }
}
