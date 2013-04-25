#pragma once

#include "Examples.hpp"

namespace ANN {
/**
 * Performs cross validation to get an overall idea of the error
 * in a given topology.
 *
 * @param examples Set of examples to use. Expects an
 * _un-transformed_ set
 *
 * @param topology Neural-net topology to create
 *
 * @param bins How many sets to split the data into (10 is reasonable)
 *
 * @return Average error (un-transformed)
 */
double crossValidation(const std::vector<Example>& examples,
                       const std::vector<unsigned>& topology, unsigned bins);
}
