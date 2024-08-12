#ifndef LEARNING_RATE_H
#define LEARNING_RATE_H

#include <math.h>
#include <vector>
#include <functional>

enum LearningRateType {
    CONSTANT = 0,
    EXPONENTIAL = 1,
    ADAM = 2
};

class LearningRate {
private:
    LearningRateType learningRateType;

    // Constant/Exponential: Initial learning rate - Adam: alpha
    double LRP1;

    // Exponential: Drop - Adam: beta1
    double LRP2;

    // Eponential: Epocs before drop - Adam: beta2
    double LRP3;

    // Adam: epsilon
    double LRP4;

    // Vector mapping learning rate type to calculating function
    std::vector<std::function<double(const int)>> learningTypeToFunc;

public:
    LearningRate(const LearningRateType _learningRateType, const double _lrp1, const double _lrp2, const double _lrp3, const double _lrp4);

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

    /**
     * @return the learning rate type
     */
    LearningRateType getLearningRateType();

    /** Accessor for alpha - used in Adam
     * @return alpha
     */
    double getAlpha();

    /** Accessor for beta1 - used in Adam
     * @return beta1
     */
    double getBeta1();

    /** Accessor for beta2 - used in Adam
     * @return beta2
     */
    double getBeta2();

    /** Accessor for epsilon - used in Adam
     * @return epsilon
     */
    double getEpsilon();
};

#endif