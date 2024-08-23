#include "LSVM.h"

LSVM::LSVM(const std::vector<std::vector<double>> _trainingData, const std::vector<int> _trainingLabels, const std::vector<std::vector<double>> _validationData, const std::vector<int> _validationLabels, const int _checkValidationSamplesEveryX, const int _numValidationLossIncBefStop, const int _numEpocs, LearningRate _learningRate, const double _indivInfluence, const double _num_samples_minibatch)
    : TRAINING_DATA(_trainingData),
    TRAINING_LABELS(_trainingLabels),
    VALIDATION_DATA(_validationData),
    VALIDATION_LABELS(_validationLabels),
    CHECK_VALIDATION_SAMPLES_EVERY_X(_checkValidationSamplesEveryX),
    NUM_VALIDATION_LOSS_INC_BEF_STOP(_numValidationLossIncBefStop),
    NUM_EPOCS(_numEpocs),
    learningRate(_learningRate),
    INDIV_INFLUENCE(_indivInfluence),
    NUM_DATA_POINTS(_trainingData.size()),
    NUM_SAMPLES_MINIBATCH(_num_samples_minibatch)
 {
    validateData();

    DIMENSION = TRAINING_DATA[0].size();

    // Init normal vector with length of the dimension of data
    normalVector = std::vector<double>(DIMENSION, 1);
}

void LSVM::validateData() {
    if (TRAINING_DATA.size() == 0)
        throw CustomException("No training data");
    if (TRAINING_DATA[0].size() == 0)
        throw CustomException("Training data has 0 dimension");
    if (TRAINING_DATA.size() != TRAINING_LABELS.size())
        throw CustomException("Training data size is different than label size");

    if (VALIDATION_DATA.size() != VALIDATION_LABELS.size())
        throw CustomException("Validation data size is different than label size");

    const int dimension = TRAINING_DATA[0].size();
    for (int i = 1; i < TRAINING_DATA.size(); ++i)
        if (dimension != TRAINING_DATA[i].size())
            throw CustomException("Training data does not all have the same dimension");
    for (int i = 0; i < VALIDATION_DATA.size(); ++i)
        if (dimension != VALIDATION_DATA[i].size())
            throw CustomException("Validation data does not all have the same dimension as the training data");

    if (NUM_SAMPLES_MINIBATCH > NUM_DATA_POINTS)
        throw CustomException("Number of samples in minibatch gradient descent cannot be greater than total number of training data points");
}

std::vector<double> LSVM::getDistancesFromCurrentDB() {
    // At the end of this function, will hold the distance from each point to the current decision boundary, or 0 if not a support vector
    std::vector<double> distances(NUM_DATA_POINTS, 0);

    // Distance from the current decision boundary for the current data point
    double distance;
    for (int i = 0; i < NUM_DATA_POINTS; ++i) {
        // Get the distance from margin
        distance = TRAINING_LABELS[i] * inner_product(TRAINING_DATA[i], normalVector) - 1;

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
        dNormalVector += normalVector - ((distances[minibatchCounter] != 0) * INDIV_INFLUENCE * TRAINING_LABELS[minibatchCounter]) * TRAINING_DATA[minibatchCounter];
    }

    dNormalVector /= NUM_SAMPLES_MINIBATCH;

    return std::make_pair(currentCost, dNormalVector);
}

void LSVM::trainWithNonParameterDependentLearningRate(const bool print, const int printEveryX) {
    // keeps track of which samples to indclude in minibatch gradient descent
    int minibatchCounter = 0;

    // Keeps track of how many times in a row the misclassification error has gone up
    int misclassIncCounter = 0;

    // The missclassification error of the previous epoc
    int prevMisclas = INT_MAX;

    // Keeps track if the misclassification error increased the last round
    bool prevMisclassInc = false;

    // The current cost and the gradient
    std::pair<double, std::vector<double>> costGradient;

    for(int epoc = 1; epoc <= NUM_EPOCS; ++epoc) {
        costGradient = getCostGradient(minibatchCounter);

        normalVector -= learningRate.getLearningRate(epoc) * costGradient.second;

        if (print && epoc % printEveryX == 0)
            std::cout << epoc << ": " << costGradient.first << std::endl;

        // Print out the current cost
        if (print && epoc % printEveryX == 0)
            std::cout << epoc << ": " << costGradient.first << std::endl;

        // Check validation samples to see if training should stop early
        if ((epoc % CHECK_VALIDATION_SAMPLES_EVERY_X || prevMisclassInc) && misclassError(VALIDATION_LABELS, predictLabels(VALIDATION_DATA)) > prevMisclas) {
            if (++misclassIncCounter >= NUM_VALIDATION_LOSS_INC_BEF_STOP)
                break;
            prevMisclassInc = true;
        } else
            prevMisclassInc = false;
    }
}

void LSVM::trainWithParameterDependentLearningRate(const bool print, const int printEveryX) {
    // keeps track of which samples to indclude in minibatch gradient descent
    int minibatchCounter = 0;

    // Keeps track of how many times in a row the misclassification error has gone up
    int misclassIncCounter = 0;

    // The missclassification error of the previous epoc
    int prevMisclas = INT_MAX;

    // Keeps track if the misclassification error increased the last round
    bool prevMisclassInc = false;

    // The current cost and the gradient
    std::pair<double, std::vector<double>> costGradient;

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

        // Print out the current cost
        if (print && epoc % printEveryX == 0)
            std::cout << epoc << ": " << costGradient.first << std::endl;

        // Check validation samples to see if training should stop early
        if ((epoc % CHECK_VALIDATION_SAMPLES_EVERY_X || prevMisclassInc) && misclassError(VALIDATION_LABELS, predictLabels(VALIDATION_DATA)) > prevMisclas) {
            if (++misclassIncCounter >= NUM_VALIDATION_LOSS_INC_BEF_STOP)
                break;
            prevMisclassInc = true;
        } else
            prevMisclassInc = false;
    }
}

void LSVM::train(const bool print, const int printEveryX) {
    if (learningRate.getLearningRateType() == ADAM) {
        trainWithParameterDependentLearningRate(print, printEveryX);
    } else {
        trainWithNonParameterDependentLearningRate(print, printEveryX);
    }
}

double LSVM::misclassError(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels) {
    if (trueLabels.size() != predictedLabels.size())
        throw CustomException("Cannot get misclassification error: trueLabels and predictedLabels do not have the same size");

    double misclassError = 0;
    for (int i = 0; i < trueLabels.size(); ++i) {
        misclassError += (trueLabels[i] != predictedLabels[i]);
    }
    misclassError = misclassError / trueLabels.size();

    return misclassError;
}

std::vector<int> LSVM::predictLabels(const std::vector<std::vector<double>>& dataSet) {
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