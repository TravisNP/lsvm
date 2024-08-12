#include "matrix_operations.h"

double inner_product(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() == 0 || rhs.size() == 0)
        throw CustomException("Cannot take inner product: one of the matrices has no data (0 rows or columns)");
    if (lhs.size() != rhs.size())
        throw CustomException("Cannot take inner product: matrices are not of compatible sizes");

    double sum = 0;
    for (int i = 0; i < lhs.size(); ++i)
        sum += lhs[i] * rhs[i];
    return sum;
}

std::vector<double> cross_product(const std::vector<std::vector<double>>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() == 0 || lhs[0].size() == 0 || rhs.size() == 0)
        throw CustomException("Cannot take cross product: one of the matrices has no data (0 rows or columns)");
    if (lhs[0].size() != rhs.size())
        throw CustomException("Cannot take cross product: matrices are not of compatible sizes");

    std::vector<double> crossProduct(lhs.size(), 0);
    for (int row = 0; row < lhs.size(); ++row) {
        for (int i = 0; i < rhs.size(); ++i)
            crossProduct[row] += lhs[row][i] * rhs[i];
    }

    return crossProduct;
}