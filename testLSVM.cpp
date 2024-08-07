#include "LSVM.h"
#include <iostream>

#define NUM_EPOCS 10
#define LEARNING_RATE .001
#define INDIV_INFLUENCE 30

int main() {
    std::vector<std::vector<double>> data = {{0, 0}, {10, 10}};
    std::vector<int> labels = {-1, 1};
    LSVM* model;
    model = new LSVM(data, labels, NUM_EPOCS, LEARNING_RATE, INDIV_INFLUENCE);

    return 0;
}