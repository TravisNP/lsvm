#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include "dependencies/LSVM.h"
#include "dependencies/gnuplot.h"

// Number of epocs
#define NUM_EPOCS 10'000

// Learning rate
#define LEARNING_RATE .001

// Hyperparameter
#define INDIV_INFLUENCE 30

// How many training samples to generate
#define NUM_TRAINING_SAMPLES 1000

// How many epocs to wait to print cost of current decision boundary during training
#define PRINT_EVERY_X 10

// How many samples to test the model on
#define NUM_TEST_SAMPLES 1000

void initData(std::vector<std::vector<double>>& data, std::vector<int>& labels, const int numberSamples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> x1(7, 1);
    std::normal_distribution<double> y1(7, 1);
    std::normal_distribution<double> xN1(1, 1);
    std::normal_distribution<double> yN1(1, 1);

    std::vector<std::pair<std::vector<double>, int>> dataLabel;

    std::vector<double> generatedPoint;
    for (int i = 0; i < numberSamples/2; ++i) {
        generatedPoint = {1, x1(gen), y1(gen)};
        dataLabel.emplace_back(generatedPoint, 1);
    }

    for (int i = 0; i < numberSamples/2; ++i) {
        generatedPoint = {1, xN1(gen), yN1(gen)};
        dataLabel.emplace_back(generatedPoint, -1);
    }

    shuffle(dataLabel.begin(), dataLabel.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
    for (int i = 0; i < numberSamples; ++i) {
        data.push_back(dataLabel[i].first);
        labels.push_back(dataLabel[i].second);
    }
}

int main() {
    std::vector<std::vector<double>> trainingData;
    std::vector<int> trainingLabels;
    initData(trainingData, trainingLabels, NUM_TRAINING_SAMPLES);

    // The Linear Support Vector Machine Object
    LSVM* model;
    try {
        model = new LSVM(trainingData, trainingLabels, NUM_EPOCS, LEARNING_RATE, INDIV_INFLUENCE);
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
    std::cout << "Decision Boundary: (" << model->getNormalVector()[0] << " " << model->getNormalVector()[1] << "," << model->getNormalVector()[2] << ")" << std::endl;

    std::vector<std::vector<double>> testData;
    std::vector<int> testLabels;
    initData(testData, testLabels, NUM_TEST_SAMPLES);

    // The predicted labels by the model
    std::vector<int> predictedLabels;

    // Predict the label(s)
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

    // Plot the training data and the decision boundary
    plotData(trainingData, trainingLabels, model->getNormalVector());

    delete model;
    return 0;
}