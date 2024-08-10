#ifndef LEARNING_RATE_H
#define LEARNING_RATE_H

#include <math.h>
#include <vector>
#include <functional>

enum LearningRateType {
    CONSTANT = 0,
    EXPONENTIAL = 1
};

class LearningRate {
private:
    LearningRateType learningRateType;

    // Initial learning rate
    double INIT_LEARNING_RATE;

    // Multiplier to drop by
    double DROP;

    // Amount of epocs before every drop
    int EPOC_DROP;

    // Vector mapping learning rate type to calculating function
    std::vector<std::function<double(const int)>> learningTypeToFunc;

public:
    LearningRate(const LearningRateType _learningRateType, const double _initLearningRate, const double _drop, const int _epocdrop);

    /**
     * Chooses the correct learning rate to return
     * @return the learning rate
     */
    double getLearningRate(const int epoc);

    /** Calculates the learning rate
     * @param epoc the current epoc
     * @return the learning rate
    */
    double getExponentialLearningRate(const int epoc);

    /**
     * @return the learning rate
     */
    double getConstantLearningRate(const int epoc);
};

#endif