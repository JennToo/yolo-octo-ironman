#include "Examples.hpp"

#include <limits>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <cassert>

namespace ANN {
    Transformation getTransformation(const std::vector<Example>& examples) {
        Transformation toRet;

        // Find offsets and spans for each output slot
        std::size_t outputCount = examples[0].output.values.size();
        for(std::size_t outi = 0; outi < outputCount; outi++) {
            double min = std::numeric_limits<double>::infinity();
            double max = -min;

            // Search each example's output at the current slot
            for(std::size_t exi = 0; exi < examples.size(); exi++) {
                const Example& example = examples[exi];
                double val = example.output.values[outi];
                if(val > max) {
                    max = val;
                }

                if(val < min) {
                    min = val;
                }
            }

            toRet.offset.push_back(min);
            toRet.span.push_back(max - min);
        }

        return toRet;
    }

    void applyTransformation(Example& ex, const Transformation& trans, bool inverse) {
        std::size_t outputSize = ex.output.values.size();
        for(std::size_t i = 0; i < outputSize; i++) {
            double newval = ex.output.values[i];
            if(!inverse) {
                newval = (newval - trans.offset[i]) / trans.span[i];
                assert(newval <= 1.0 + 0.00001);
            } else {
                newval = newval * trans.span[i] + trans.offset[i];
            }
            ex.output.values[i] = newval;
        }
    }

    void applyTransformation(std::vector<Example>& examples, const Transformation& trans, bool inverse) {
        for(std::size_t i = 0; i < examples.size(); i++) {
            applyTransformation(examples[i], trans, inverse);
        }
    }

    void loadExamplesFromFile(const std::string& file, std::vector<Example>& examples,
                              std::size_t inputs, std::size_t outputs) {
        std::ifstream inFile(file.c_str());
        if(inFile.fail()) {
            std::cerr << "Could not open examples file: " << file << std::endl;
            exit(1);
        }

        std::vector<double> vals;

        while(!inFile.eof()) {
            std::string line;
            getline(inFile, line, '\n');

            if(line.length() != 0) {
                std::istringstream in(line);
                double val;
                while(in >> val) {
                    vals.push_back(val);
                    if(in.peek() == ',') {
                        in.ignore();
                    }
                }
            }

            if(vals.size() % (inputs*outputs) != 0) {
                std::cerr << "Malformed input line in file: " << file
                          << " line: " << line << std::endl;
                exit(1);
            }
        }

    }
}
