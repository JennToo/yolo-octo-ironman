TEMPLATE = lib
CONFIG = staticlib warn_on debug console
OBJECTS_DIR = ../build
MOC_DIR = ../build
SOURCES = NeuralNet.cpp Connection.cpp Examples.cpp CrossValidation.cpp DiscreteClassifier.cpp
HEADERS = NeuralNet.hpp Connection.hpp Examples.hpp Neuron.hpp Util.hpp CrossValidation.hpp DiscreteClassifier.hpp
DESTDIR = ../bin
QMAKE_CXXFLAGS += -std=c++0x