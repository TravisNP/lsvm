#include <iostream>
#include <random>
#include <algorithm>
#include <chrono>

#include "dependencies/LSVM.h"

#define NUM_EPOCS 1'000
#define LEARNING_RATE .001
#define INDIV_INFLUENCE 30

#define NUM_SAMPLES 10

void initData(std::vector<std::vector<double>>& data, std::vector<int>& labels) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> x1(3, 1);
    std::normal_distribution<double> y1(1, 1);
    std::normal_distribution<double> xN1(9, 1);
    std::normal_distribution<double> yN1(7, 1);

    std::vector<std::pair<std::vector<double>, int>> dataLabel;

    std::vector<double> generatedPoint;
    for (int i = 0; i < NUM_SAMPLES/2; ++i) {
        generatedPoint = {x1(gen), y1(gen)};
        dataLabel.emplace_back(generatedPoint, 1);
    }

    for (int i = 0; i < NUM_SAMPLES/2; ++i) {
        generatedPoint = {xN1(gen), yN1(gen)};
        dataLabel.emplace_back(generatedPoint, -1);
    }

    shuffle(dataLabel.begin(), dataLabel.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        data.push_back(dataLabel[i].first);
        labels.push_back(dataLabel[i].second);
    }
}

int main() {
    std::vector<std::vector<double>> data;
    std::vector<int> labels;
    initData(data, labels);

    for (int i = 0; i < NUM_SAMPLES; ++i)
        std::cout << "(" << data[i][0] << "," << data[i][1] << "): " << labels[i] << std::endl;

    LSVM* model;
    try {
        model = new LSVM(data, labels, NUM_EPOCS, LEARNING_RATE, INDIV_INFLUENCE);
    } catch (CustomException& e) {
        std::cout << "Error in creation of model - " << e.getMessage() << std::endl;
        delete model;
        return -1;
    }

    try {
        model->train(true, 100);
    } catch (CustomException& e) {
        std::cout << "Error in training of model - " << e.getMessage() << std::endl;
    }

    std::cout <<    "Normal Vector: (" << model->getNormalVector()[0] << "," << model->getNormalVector()[1] << ")" << std::endl;

    std::vector<int> predictedLabels;
    try {
        predictedLabels = model->predictLabels(data);
    } catch (CustomException& e) {
        std::cout << "Error in predicting labels - " << e.getMessage() << std::endl;
    }

    double accuracy = 0;
    for (int i = 0; i < data.size(); ++i) {
        accuracy += (labels[i] == predictedLabels[i]);
    }
    accuracy = accuracy * 100 / data.size();
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;

    delete model;
    return 0;
}