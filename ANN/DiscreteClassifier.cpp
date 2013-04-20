#include "DiscreteClassifier.hpp"
#include "Util.hpp"

namespace ANN {
    std::size_t DiscreteClassifier::getClassificationIndex(double continuousValue,
                                                          ClassifierMethod method) const {
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

    double DiscreteClassifier::getIndexValue(std::size_t index) const {
        return values[index];
    }
}
