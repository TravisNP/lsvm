#ifndef LSVM_H
#define LSVM_H

#include <vector>
#include <iostream>
#include <numeric>
#include <math.h>
#include <limits.h>

#include "overload.h"
#include "learning_rate.h"
#include "data.h"

class LSVM {
private:
    // The data to train the model on
    std::vector<std::vector<double>> DATA;

    // The labels to train the model on
    std::vector<int> LABELS;

    // Number of epocs used during training
    int NUM_EPOCS;

    // Hyperparameter
    LearningRate learningRate;

    // Hyperparameter
    double INDIV_INFLUENCE;

    std::vector<double> normalVector;

    // Dimension of the data
    int DIMENSION;

    // Number of data points to generate
    int NUM_DATA_POINTS;

    // Threshold for the cost difference change to end training early
    double COST_PERCENTAGE_THRESHOLD;

    // Number of times the cost difference change needs to be below the threshold to stop training early
    double NUM_COST_BELOW_THRESHOLD;

    // The amount of samples in each minibatch gradient descent
    double NUM_SAMPLES_MINIBATCH;

    /** Validates the data
     * @throws an error if the dataset is not valid
     */
    void validateData();

    /** Calculates the distance of each point from the current decision boundary
     * @return the distance of each point form the current decision boundary or 0 if not a support vector (outside margin)
     */
    std::vector<double> getDistancesFromCurrentDB();

    /** Calculates the cost gradients to do gradient descent
     * @param minibatchCounter keeps track of which samples to indclude in minibatch gradient descent
     * @return the cost and the gradient
     */
    std::pair<double, std::vector<double>> getCostGradient(int& minibatchCounter);

public:
    LSVM(const std::vector<std::vector<double>> _data, const std::vector<int> _labels, const int _numEpocs, LearningRate learningRate, const double _indivInfluence, const double _cost_percentage_threshold, const double _num_cost_below_threshold, const double _num_samples_minibatch);

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

    /** Accessor for the dimension
     * @return the dimension
     */
    int getDimension();
};

#endif