#ifndef LSVM_H
#define LSVM_H

#include <vector>
#include <iostream>
#include <numeric>
#include <math.h>

#include "overload.h"

class LSVM {
private:
    std::vector<std::vector<double>> DATA;

    std::vector<int> LABELS;

    int NUM_EPOCS;

    double LEARNING_RATE;

    double INDIV_INFLUENCE;

    std::vector<double> normalVector;

    int DIMENSION;

    int NUM_DATA_POINTS;

    // Throws an error if the dataset is not valid
    void validateData();

    /** Calculates the distance of each point from the current decision boundary
     * @return the distance of each point form the current decision boundary or 0 if not a support vector (outside margin)
     */
    std::vector<double> getDistancesFromCurrentDB();

    /** Calculates the cost gradients to do gradient descent
     * @return the cost (no actual use besides printing to terminal) and the gradient
     */
    std::pair<int, std::vector<double>> getCostGradient();

public:
    LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, const double _learningRate, const double _indivInfluence);

    /** Fits the model to the data with gradient descent
     * @param printEveryX prints the cost every X iterations
     */
    void train(const bool print = false, const int printEveryX = 1000);

    /** Predicts the labels of the inputted dataset
     * @param dataSet the dataset to predict the labels of
     * @return the predicted labels
     */
    std::vector<int> predictLabels(const std::vector<std::vector<double>> dataSet);

    /** Accessor for the normalVector variable
     * @return the normal vector
     */
    std::vector<double> getNormalVector();
};

#endif