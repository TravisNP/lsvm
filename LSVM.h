#ifndef LSVM_H
#define LSVM_H

#include <vector>

class LSVM {
private:
    std::vector<std::vector<double>> data;

    std::vector<int> labels;

    int numEpocs;

    double learningRate;

    double indivInfluence;

    std::vector<double> normalVector;

public:
    LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, const double _learningRate, const double _indivInfluence);
};

#endif