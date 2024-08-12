#include <iostream>
#include <algorithm>

#include "dependencies/LSVM.h"
#include "dependencies/learning_rate.h"
#include "dependencies/gnuplot.h"

// Number of epocs
#define NUM_EPOCS 200

// Hyperparameter
#define INDIV_INFLUENCE 30

// How many training samples to generate
#define NUM_TRAINING_SAMPLES 10'000

// How many epocs to wait to print cost of current decision boundary during training
#define PRINT_EVERY_X 10

// How many samples to test the model on
#define NUM_TEST_SAMPLES 1000

// The amount of samples in each minibatch gradient descent
#define NUM_SAMPLES_MINIBATCH 64

// Learning rate type
#define LEARNING_RATE_TYPE ADAM

// // Decent Exponential Parameter Values
// // Initial learning rate
// #define LRP1 1
// // Drop
// #define LRP2 .5
// // Epocs before drop
// #define LRP3 50

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
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingLabels;
    init2dData(trainingData, trainingLabels, NUM_TRAINING_SAMPLES);
    // load2dData("data.dat", trainingData, trainingLabels);

    // The Linear Support Vector Machine Object
    LSVM* model;
    try {
        LearningRate learningRate = LearningRate(LEARNING_RATE_TYPE, LRP1, LRP2, LRP3, LRP4);
        model = new LSVM(trainingData, trainingLabels, NUM_EPOCS, learningRate, INDIV_INFLUENCE, NUM_SAMPLES_MINIBATCH);
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

    // Print out the vector for the decision boundary
    printDecisionBoundary(model);

    // Create the test data and labels
    std::vector<std::vector<double>> testData;
    std::vector<int> testLabels;
    init2dData(testData, testLabels, NUM_TEST_SAMPLES);

    // Predict the label(s)
    // The predicted labels by the model
    std::vector<int> predictedLabels;
    try {
        predictedLabels = model->predictLabels(testData);
    } catch (CustomException& e) {
        std::cout << "Error in predicting labels - " << e.getMessage() << std::endl;
    }

    // Calculate the arruracy of the model
    double accuracy = 0;
    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        accuracy += (testLabels[i] == predictedLabels[i]);
    }
    accuracy = accuracy * 100 / NUM_TEST_SAMPLES;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    // Save and plot the data
    saveData(trainingData, trainingLabels, model->getNormalVector(), "data.dat");
    plot2dData("data.dat");

    delete model;
    return 0;
}