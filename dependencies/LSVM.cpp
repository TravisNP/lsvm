#include "LSVM.h"

LSVM::LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, const double _learningRate, const double _indivInfluence)
    : DATA(_data),
    LABELS(_labels),
    NUM_EPOCS(_numEpocs),
    LEARNING_RATE(_learningRate),
    INDIV_INFLUENCE(_indivInfluence),
    NUM_DATA_POINTS(_data.size())
 {
    validateData();

    DIMENSION = DATA[0].size();

    // Init normal vector with length of the dimension of data
    normalVector = std::vector<double>(DIMENSION, 0);
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
        distance = LABELS[i] * std::inner_product(DATA[i].begin(), DATA[i].end(), normalVector.begin(), 0) - 1;

        // If distance is less than 0, then datapoint is a support vector and the distance matters
        // If distance greater than 0, outside of margin so not a support vector
        if (distance <= 0)
            distances[i] = distance;
    }

    return distances;
}

std::pair<int, std::vector<double>> LSVM::getCostGradient() {
    // The distance from each point to the current decision boundary, or 0 if not a support vector
    std::vector<double> distances = getDistancesFromCurrentDB();

    // Current cost of the support vectors
    double currentCost = 0.5 * std::inner_product(normalVector.begin(), normalVector.end(), normalVector.begin(), 0) - currentCost * std::reduce(distances.begin(), distances.end());

    // Gradient
    std::vector<double> dNormalVector(DIMENSION, 0);

    for (int i = 0; i < NUM_DATA_POINTS; ++i) {
        if (distances[i] != 0)
            dNormalVector -= INDIV_INFLUENCE * LABELS[i] * DATA[i];

        dNormalVector += normalVector;
    }

    return std::make_pair(currentCost, dNormalVector);
}