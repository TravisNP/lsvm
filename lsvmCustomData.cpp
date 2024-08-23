#include <iostream>
#include <algorithm>

#include "dependencies/LSVM.h"
#include "dependencies/learning_rate.h"
#include "dependencies/gnuplot.h"

// Number of epocs
#define NUM_EPOCS 250

// How often to check validation samples for early stopping
#define CHECK_VALIDATION_SAMPLES_EVERY_X 5

// How many times the validation loss can increase before early stopping
#define NUM_VALIDATION_LOSS_INC_BEF_STOP 5


// The amount of samples in each minibatch gradient descent
#define NUM_SAMPLES_MINIBATCH 50

// Hyperparameter
#define INDIV_INFLUENCE 30

// Learning rate type
#define LEARNING_RATE_TYPE ADAM

// How many epocs to wait to print cost of current decision boundary during training
#define PRINT_EVERY_X 100

// // Decent Exponential Parameter Values
// // Initial learning rate
// #define LRP1 .001
// // Drop
// #define LRP2 .5
// // Epocs before drop
// #define LRP3 50
// // Unused
// #define LRP4 0

// Decent Adam parameter values
// Adam: alpha
#define LRP1 .02
// Adam: beta1
#define LRP2 .8
// Adam: beta2
#define LRP3 .999
// Adam: epsilon
#define LRP4 pow(10, -8)

void printDecisionBoundary(LSVM* model) {
    std::vector<double> decisionBoundary = model->getNormalVector();
    std::cout << "Decision Boundary: (";
    for (int i = 0; i < model->getDimension() - 1; ++i)
        std::cout << decisionBoundary[i] << ",";
    std::cout << decisionBoundary.back() << ")" << std::endl;
}

int main() {
    // Load training data and lables
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingLabels;
    loadCustomData("trainingData.dat", trainingData, trainingLabels, 5);

    // Load validation data and labels
    std::vector<std::vector<double>> validationData;
    std::vector<int> validationLabels;
    loadCustomData("validationData.dat", validationData, validationLabels, 5);

    // The Linear Support Vector Machine Object
    LSVM* model;
    try {
        LearningRate learningRate = LearningRate(LEARNING_RATE_TYPE, LRP1, LRP2, LRP3, LRP4);
        model = new LSVM(trainingData, trainingLabels, validationData, validationLabels, CHECK_VALIDATION_SAMPLES_EVERY_X, NUM_VALIDATION_LOSS_INC_BEF_STOP, NUM_EPOCS, learningRate, INDIV_INFLUENCE, NUM_SAMPLES_MINIBATCH);
    } catch (CustomException& e) {
        std::cout << "Error in creation of model - " << e.getMessage() << std::endl;
        delete model;
        return -1;
    }

    // Train the data
    try {
        model->train(true, PRINT_EVERY_X);
    } catch (CustomException& e) {
        std::cout << "Error in training of model - " << e.getMessage() << std::endl;
    }

    // Load the test data and labels
    std::vector<std::vector<double>> testData;
    std::vector<int> testLabels;
    loadCustomData("testData.dat", testData, testLabels, 5);

    // Predict the label(s)
    // The predicted labels by the model
    std::vector<int> predictedLabels;
    try {
        predictedLabels = model->predictLabels(testData);
    } catch (CustomException& e) {
        std::cout << "Error in predicting labels - " << e.getMessage() << std::endl;
    }

    // Calculate the arruracy of the model
    double misclassError = model->misclassError(testLabels, predictedLabels);
    std::cout << "Accuracy: " << 1 - misclassError << std::endl;
    std::cout << "Num Wrong: " << misclassError * testData.size() << std::endl;
    std::cout << "Decision Boundary: " << model->getNormalVector() << std::endl;

    delete model;
    return 0;
}