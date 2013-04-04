TEMPLATE = subdirs
CONFIG += ordered
SUBDIRS = ANN ANN_Driver
ANN_Driver.depends = ANN
QMAKE_CXXFLAGS += -std=c++0x
