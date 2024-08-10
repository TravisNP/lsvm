#include "LSVM.h"

LSVM::LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, const double _learningRate, const double _indivInfluence, const double _cost_percentage_threshold, const double _num_cost_below_threshold)
    : DATA(_data),
    LABELS(_labels),
    NUM_EPOCS(_numEpocs),
    LEARNING_RATE(_learningRate),
    INDIV_INFLUENCE(_indivInfluence),
    NUM_DATA_POINTS(_data.size()),
    COST_PERCENTAGE_THRESHOLD(_cost_percentage_threshold),
    NUM_COST_BELOW_THRESHOLD(_num_cost_below_threshold)
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
}

std::vector<double> LSVM::getDistancesFromCurrentDB() {
    // At the end of this function, will hold the distance from each point to the current decision boundary, or 0 if not a support vector
    std::vector<double> distances(NUM_DATA_POINTS, 0);

    // Distance from the current decision boundary for the current data point
    double distance;
    for (int i = 0; i < NUM_DATA_POINTS; ++i) {
        // Get the distance from margin
        distance = LABELS[i] * (DATA[i] * normalVector) - 1;

        // If distance is less than 0, then datapoint is a support vector and the distance matters
        // If distance greater than 0, outside of margin so not a support vector
        if (distance <= 0)
            distances[i] = distance;
    }

    return distances;
}

std::pair<double, std::vector<double>> LSVM::getCostGradient() {
    // The distance from each point to the current decision boundary, or 0 if not a support vector
    std::vector<double> distances = getDistancesFromCurrentDB();

    // Current cost of the support vectors
    double currentCost = 0.5 * (normalVector * normalVector) - INDIV_INFLUENCE * std::reduce(distances.begin(), distances.end());

    // Gradient
    std::vector<double> dNormalVector(DIMENSION, 0);

    for (int i = 0; i < NUM_DATA_POINTS; ++i) {
        dNormalVector += normalVector - ((distances[i] != 0) * INDIV_INFLUENCE * LABELS[i]) * DATA[i];
    }

    dNormalVector /= NUM_DATA_POINTS;

    return std::make_pair(currentCost, dNormalVector);
}

void LSVM::train(const bool print, const int printEveryX) {
    std::pair<double, std::vector<double>> costGradient = getCostGradient();
    normalVector -= LEARNING_RATE * costGradient.second;
    if (print && printEveryX == 1)
        std::cout << 1 << ": " << costGradient.first << std::endl;

    double prevCost = costGradient.first;
    int breakCounter = 0;

    for(int epoc = 2; epoc <= NUM_EPOCS; ++epoc) {
        costGradient = getCostGradient();
        if (abs(costGradient.first/prevCost - 1) < COST_PERCENTAGE_THRESHOLD && ++breakCounter < NUM_COST_BELOW_THRESHOLD)
            break;
        else
            breakCounter = 0;

        prevCost = costGradient.first;

        normalVector -= LEARNING_RATE * costGradient.second;
        if (print && epoc % printEveryX == 0)
            std::cout << epoc << ": " << costGradient.first << std::endl;
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
    std::vector<double> crossProduct = dataSet * normalVector;

    // The predicted labels
    std::vector<int> signs(dataSet.size(), 0);
    for (int i = 0; i < dataSet.size(); ++i)
        signs[i] = copysign(1, crossProduct[i]);

    return signs;
}

std::vector<double> LSVM::getNormalVector() {
    return normalVector;
}