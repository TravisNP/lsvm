#include <iostream>

#include "dependencies/LSVM.h"

#define NUM_EPOCS 10
#define LEARNING_RATE .001
#define INDIV_INFLUENCE 30

int main() {
    std::vector<std::vector<double>> data = {{0, 0}, {10, 10}};
    std::vector<int> labels = {-1, 1};
    LSVM* model;
    try {
        model = new LSVM(data, labels, NUM_EPOCS, LEARNING_RATE, INDIV_INFLUENCE);
    } catch (CustomException& e) {
        std::cout << "Error in creation of model: " << e.getMessage() << std::endl;
        delete model;
        return -1;
    }

    delete model;
    return 0;
}