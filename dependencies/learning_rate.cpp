#include "learning_rate.h"

LearningRate::LearningRate(const LearningRateType _learningRateType, const double _lrp1, const double _lrp2, const double _lrp3, const double _lrp4) :
    learningRateType(_learningRateType),
    LRP1(_lrp1),
    LRP2(_lrp2),
    LRP3(_lrp3),
    LRP4(_lrp4)
{
    learningTypeToFunc.push_back([this](const int epoc) { return this->getConstantLearningRate(epoc); });
    learningTypeToFunc.push_back([this](const int epoc) { return this->getExponentialLearningRate(epoc); });
}

double LearningRate::getLearningRate(const int epoc) {
    learningTypeToFunc[learningRateType](epoc);
}

double LearningRate::getExponentialLearningRate(const int epoc) {
    return LRP1 * pow(LRP2, floor((1 + epoc) / LRP3));
}

double LearningRate::getConstantLearningRate(const int epoc) {
    return LRP1;
}

LearningRateType LearningRate::getLearningRateType() {
    return learningRateType;
}

double LearningRate::getAlpha() {
    return LRP1;
}

double LearningRate::getBeta1() {
    return LRP2;
}

double LearningRate::getBeta2() {
    return LRP3;
}

double LearningRate::getEpsilon() {
    return LRP4;
}