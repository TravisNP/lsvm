#include <iostream>
#include <algorithm>

#include "dependencies/LSVM.h"
#include "dependencies/learning_rate.h"
#include "dependencies/gnuplot.h"

// Number of epocs
#define NUM_EPOCS 100'000

// Threshold percentage on costs for stopping training early
#define COST_PERCENTAGE_THRESHOLD .0001

// Number of times the cost difference change needs to be below the threshold to stop training early
#define NUM_COST_BELOW_THRESHOLD 30

// Hyperparameter
#define INDIV_INFLUENCE 30

// How many training samples to generate
#define NUM_TRAINING_SAMPLES 1000

// How many epocs to wait to print cost of current decision boundary during training
#define PRINT_EVERY_X 1000

// How many samples to test the model on
#define NUM_TEST_SAMPLES 1000

// The amount of samples in each minibatch gradient descent
#define NUM_SAMPLES_MINIBATCH 1000

// Learning rate related
// Learning rate
#define INITIAL_LEARNING_RATE .001
// Drop amount
#define DROP .5
// Number of epocs before drop
#define EPOC_DROP 10

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
    // init2dData(trainingData, trainingLabels, NUM_TRAINING_SAMPLES);
    load2dData("data.dat", trainingData, trainingLabels);

    // The Linear Support Vector Machine Object
    LSVM* model;
    try {
        LearningRate learningRate = LearningRate(CONSTANT, INITIAL_LEARNING_RATE, DROP, EPOC_DROP);
        model = new LSVM(trainingData, trainingLabels, NUM_EPOCS, learningRate, INDIV_INFLUENCE, COST_PERCENTAGE_THRESHOLD, NUM_COST_BELOW_THRESHOLD, NUM_SAMPLES_MINIBATCH);
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