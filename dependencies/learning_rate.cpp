#include "learning_rate.h"

LearningRate::LearningRate(const LearningRateType _learningRateType, const double _initLearningRate, const double _drop, const int _epocDrop, const int _dimension) :
    learningRateType(_learningRateType),
    INIT_LEARNING_RATE(_initLearningRate),
    DROP(_drop),
    EPOC_DROP(_epocDrop),
    DIMENSION(_dimension)
{
    learningTypeToFunc.push_back([this](const int epoc) { return this->getConstantLearningRate(epoc); });
    learningTypeToFunc.push_back([this](const int epoc) { return this->getExponentialLearningRate(epoc); });
}

double LearningRate::getLearningRate(const int epoc) {
    learningTypeToFunc[learningRateType](epoc);
}

double LearningRate::getExponentialLearningRate(const int epoc) {
    return INIT_LEARNING_RATE * pow(DROP, floor((1 + epoc) / EPOC_DROP));
}

double LearningRate::getConstantLearningRate(const int epoc) {
    return INIT_LEARNING_RATE;
}