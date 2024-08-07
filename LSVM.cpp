#include "LSVM.h"

LSVM::LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, const double _learningRate, const double _indivInfluence)
    : data(_data),
    labels(_labels),
    numEpocs(_numEpocs),
    learningRate(_learningRate),
    indivInfluence(_indivInfluence),

    // Init normal vector with length of the dimension of data
    normalVector(std::vector<double>(_data[0].size()))
 {

}