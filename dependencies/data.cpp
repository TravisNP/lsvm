#include "data.h"

void saveData(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, const std::vector<double>& decisionBoundary, const std::string file_name) {
    std::ofstream file(file_name);
    file << decisionBoundary[0] << " " << decisionBoundary[1] << " " << decisionBoundary[2] << std::endl;
    for (int i = 0; i < data.size(); ++i) {
        file << data[i][1] << " " << data[i][2] << " " << labels[i] << std::endl;
    }
    file.close();
}

void load2dData(std::string fileName, std::vector<std::vector<double>>& data, std::vector<int>& labels) {
    std::ifstream infile(fileName);
    std::string line;
    std::vector<double> dataPoint(3, 1);

    // Skip the first line because that is the decision boundary
    std::getline(infile, line);

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string num;
        for (int i = 1; i < 3; ++i) {
            iss >> num;
            dataPoint[i] = (std::stod(num));
        }
        data.push_back(dataPoint);
        iss >> num;
        labels.push_back(std::stoi(num));
    }
}

void loadCustomData(std::string fileName, std::vector<std::vector<double>>& data, std::vector<int>& labels, const int numDataFields) {
    std::ifstream infile(fileName);
    std::string line;

    std::vector<double> dataPoint(numDataFields, 1);

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string num;
        for (int i = 1; i < numDataFields; ++i) {
            iss >> num;
            dataPoint[i] = (std::stod(num));
        }
        data.push_back(dataPoint);
        iss >> num;
        labels.push_back(std::stoi(num));
    }
}

void init2dData(std::vector<std::vector<double>>& data, std::vector<int>& labels, const int numberSamples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> x1(1, 2);
    std::normal_distribution<double> y1(1, 1);
    std::normal_distribution<double> xN1(7, 1);
    std::normal_distribution<double> yN1(7, 2);

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

void init3dData(std::vector<std::vector<double>>& data, std::vector<int>& labels, const int numberSamples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> x1(7, 1);
    std::normal_distribution<double> y1(7, 1);
    std::normal_distribution<double> z1(7, 1);
    std::normal_distribution<double> xN1(1, 1);
    std::normal_distribution<double> yN1(1, 1);
    std::normal_distribution<double> zN1(1, 1);

    std::vector<std::pair<std::vector<double>, int>> dataLabel;

    std::vector<double> generatedPoint;
    for (int i = 0; i < numberSamples/2; ++i) {
        generatedPoint = {1, x1(gen), y1(gen), z1(gen)};
        dataLabel.emplace_back(generatedPoint, 1);
    }

    for (int i = 0; i < numberSamples/2; ++i) {
        generatedPoint = {1, xN1(gen), yN1(gen), zN1(gen)};
        dataLabel.emplace_back(generatedPoint, -1);
    }

    shuffle(dataLabel.begin(), dataLabel.end(), std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count()));
    for (int i = 0; i < numberSamples; ++i) {
        data.push_back(dataLabel[i].first);
        labels.push_back(dataLabel[i].second);
    }
}