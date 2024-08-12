#ifndef LSVM_H
#define LSVM_H

#include <vector>
#include <iostream>
#include <numeric>
#include <math.h>
#include <limits.h>

#include "overload.h"
#include "matrix_operations.h"
#include "learning_rate.h"
#include "data.h"

class LSVM {
private:
    // The data to train the model on
    std::vector<std::vector<double>> TRAINING_DATA;

    // The labels to train the model on
    std::vector<int> TRAINING_LABELS;

    // The data to validate the model on
    std::vector<std::vector<double>> VALIDATION_DATA;

    // The labels to validate the model on
    std::vector<int> VALIDATION_LABELS;

    /** How often to check validation samples for early stopping
     * Note if the validation loss increases, the next one is checked regardless of this variable*/
    int CHECK_VALIDATION_SAMPLES_EVERY_X;

    // How many times the validation loss can increase before early stopping
    int NUM_VALIDATION_LOSS_INC_BEF_STOP;

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
     * @param minibatchCounter keeps track of which samples to include in minibatch gradient descent
     * @return the cost and the gradient
     */
    std::pair<double, std::vector<double>> getCostGradient(int& minibatchCounter);

    /** Fits the model to the data with gradient descent with a learning rate that is not parameter dependent (CONSTANT, EXPONENTIAL)
     * @param print true to print the cost every printEveryX iterations, false for no print
     * @param printEveryX prints the cost every X iterations
     */
    void trainWithNonParameterDependentLearningRate(const bool print, const int printEveryX);

    /** Fits the model to the data with gradient descent with a learning rate that adapts to the individual parameters (ADAM)
     * @param print true to print the cost every printEveryX iterations, false for no print
     * @param printEveryX prints the cost every X iterations
     */
    void trainWithParameterDependentLearningRate(const bool print, const int printEveryX);

public:
    LSVM(const std::vector<std::vector<double>> _trainingData, const std::vector<int> _trainingLabels, const std::vector<std::vector<double>> _validationData, const std::vector<int> _validationLabels, const int _checkValidationSamplesEveryX, const int _numValidationLossIncBefStop, const int _numEpocs, LearningRate learningRate, const double _indivInfluence, const double _num_samples_minibatch);

    /** Decides which training method to call based off the learning rate type
     * @param print true to print the cost every printEveryX iterations, false for no print
     * @param printEveryX prints the cost every X iterations
     */
    void train(const bool print = false, const int printEveryX = 1000);

    /** Predicts the labels of the inputted dataset
     * @param dataSet the dataset to predict the labels of
     * @return the predicted labels
     */
    std::vector<int> predictLabels(const std::vector<std::vector<double>>& dataSet);

    /** Gets the misclassification error of the predicted labels
     * @param trueLabels the actualy labels for the data
     * @param predictedLabels the predicted labels by the model for the data
     * @return the misclassification error
     */
    double misclassError(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels);

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