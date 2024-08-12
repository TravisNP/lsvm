#include "LSVM.h"

LSVM::LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, LearningRate _learningRate, const double _indivInfluence, const double _num_samples_minibatch)
    : DATA(_data),
    LABELS(_labels),
    NUM_EPOCS(_numEpocs),
    learningRate(_learningRate),
    INDIV_INFLUENCE(_indivInfluence),
    NUM_DATA_POINTS(_data.size()),
    NUM_SAMPLES_MINIBATCH(_num_samples_minibatch)
 {
    validateData();

    DIMENSION = DATA[0].size();

    // Init normal vector with length of the dimension of data
    normalVector = std::vector<double>(DIMENSION, 1);
}

void LSVM::validateData() {
    if (DATA.size() == 0)
        throw CustomException("No data");
    if (DATA[0].size() == 0)
        throw CustomException("Data has 0 dimension");
    if (DATA.size() != LABELS.size())
        throw CustomException("Data size is different than label size");

    const int dimension = DATA[0].size();
    for (int i = 1; i < DATA.size(); ++i)
        if (dimension != DATA[i].size())
            throw CustomException("Data does not all have the same dimension");

    if (NUM_SAMPLES_MINIBATCH > NUM_DATA_POINTS)
        throw CustomException("Number of samples in minibatch gradient descent cannot be greater than total number of data points");
}

std::vector<double> LSVM::getDistancesFromCurrentDB() {
    // At the end of this function, will hold the distance from each point to the current decision boundary, or 0 if not a support vector
    std::vector<double> distances(NUM_DATA_POINTS, 0);

    // Distance from the current decision boundary for the current data point
    double distance;
    for (int i = 0; i < NUM_DATA_POINTS; ++i) {
        // Get the distance from margin
        distance = LABELS[i] * inner_product(DATA[i], normalVector) - 1;

        // If distance is less than 0, then datapoint is a support vector and the distance matters
        // If distance greater than 0, outside of margin so not a support vector
        if (distance <= 0)
            distances[i] = distance;
    }

    return distances;
}

std::pair<double, std::vector<double>> LSVM::getCostGradient(int& minibatchCounter) {
    // The distance from each point to the current decision boundary, or 0 if not a support vector
    std::vector<double> distances = getDistancesFromCurrentDB();

    // Current cost of the support vectors
    double currentCost = 0.5 * inner_product(normalVector, normalVector) - INDIV_INFLUENCE * std::reduce(distances.begin(), distances.end());

    // Gradient
    std::vector<double> dNormalVector(DIMENSION, 0);

    for (int i = 0; i < NUM_SAMPLES_MINIBATCH; ++i) {
        if (++minibatchCounter >= NUM_DATA_POINTS)
            minibatchCounter = 0;
        dNormalVector += normalVector - ((distances[minibatchCounter] != 0) * INDIV_INFLUENCE * LABELS[minibatchCounter]) * DATA[minibatchCounter];
    }

    dNormalVector /= NUM_SAMPLES_MINIBATCH;

    return std::make_pair(currentCost, dNormalVector);
}

void LSVM::trainWithNonParameterDependentLearningRate(const bool print, const int printEveryX) {
    // keeps track of which samples to indclude in minibatch gradient descent
    int minibatchCounter = 0;

    // The current cost and the gradient
    std::pair<double, std::vector<double>> costGradient;

    // The cost for the previous iteration
    double prevCost = 1;

    // A counter to keep track of when to stop trianing early
    int breakCounter = 0;

    for(int epoc = 1; epoc <= NUM_EPOCS; ++epoc) {
        costGradient = getCostGradient(minibatchCounter);

        normalVector -= learningRate.getLearningRate(epoc) * costGradient.second;

        prevCost = costGradient.first;
        if (print && epoc % printEveryX == 0)
            std::cout << epoc << ": " << costGradient.first << std::endl;
    }
}

void LSVM::trainWithParameterDependentLearningRate(const bool print, const int printEveryX) {
    // keeps track of which samples to indclude in minibatch gradient descent
    int minibatchCounter = 0;

    // The current cost and the gradient
    std::pair<double, std::vector<double>> costGradient;

    // The cost for the previous iteration
    double prevCost = 1;

    // A counter to keep track of when to stop trianing early
    int breakCounter = 0;

    // First moment of the normal vector
    std::vector<double> firstMoment(DIMENSION, 0);

    // Second moment of the normal vector
    std::vector<double> secondMoment(DIMENSION, 0);

    // Unbiased first moment
    double firstMomentParamUnbiased;

    // Unbiased second moment
    double secondMomentParamUnbiased;

    // Beta1 to the epoc power
    double beta1EpocPower = learningRate.getBeta1();

    // Beta1 to the epoc power
    double beta2EpocPower = learningRate.getBeta2();

    for(int epoc = 1; epoc <= NUM_EPOCS; ++epoc) {
        costGradient = getCostGradient(minibatchCounter);

        // For each parameter
        for (int param = 0; param < DIMENSION; ++param) {
            // Update the first and second moments
            firstMoment[param] = learningRate.getBeta1() * firstMoment[param] + (1 - learningRate.getBeta1()) * costGradient.second[param];
            secondMoment[param] = learningRate.getBeta2() * secondMoment[param] + (1 - learningRate.getBeta2()) * pow(costGradient.second[param], 2);

            // Get the unbiased versions
            firstMomentParamUnbiased = firstMoment[param] / (1 - beta1EpocPower);
            secondMomentParamUnbiased = secondMoment[param] / (1 - beta2EpocPower);

            // Update the beta1 and beta2 powers
            beta1EpocPower *= learningRate.getBeta1();
            beta2EpocPower *= learningRate.getBeta2();

            normalVector[param] -= learningRate.getAlpha() * firstMomentParamUnbiased / (sqrt(secondMomentParamUnbiased) + learningRate.getEpsilon());
        }

        prevCost = costGradient.first;
        if (print && epoc % printEveryX == 0)
            std::cout << epoc << ": " << costGradient.first << std::endl;
    }
}

void LSVM::train(const bool print, const int printEveryX) {
    if (learningRate.getLearningRateType() == ADAM) {
        trainWithParameterDependentLearningRate(print, printEveryX);
    } else {
        trainWithNonParameterDependentLearningRate(print, printEveryX);
    }
}

std::vector<int> LSVM::predictLabels(std::vector<std::vector<double>> dataSet) {
    // Validate the dataset
    if (dataSet.size() == 0)
        throw CustomException("Cannot predict: no data");
    for (int i = 0; i < dataSet.size(); ++i)
        if (dataSet[i].size() != DIMENSION)
            throw CustomException("Cannot predict: dimension of each point" + std::to_string(i) + "does not match dimension of training data");

    // The cross product of the dataset and the decision boundary
    std::vector<double> crossProduct = cross_product(dataSet, normalVector);

    // The predicted labels
    std::vector<int> signs(dataSet.size(), 0);
    for (int i = 0; i < dataSet.size(); ++i)
        signs[i] = copysign(1, crossProduct[i]);

    return signs;
}

std::vector<double> LSVM::getNormalVector() {
    return normalVector;
}

int LSVM::getDimension() {
    return DIMENSION;
}