#include "overload.h"

std::vector<double>& operator+=(std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size())
        throw CustomException("Cannot add vectors because their lengths are not the same");

    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] += rhs[i];
    return lhs;
}

std::vector<double>& operator-=(std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size())
        throw CustomException("Cannot subtract vectors because their lengths are not the same");

    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] -= rhs[i];
    return lhs;
}

std::vector<double>& operator/=(std::vector<double>& vec, const int divisor) {
    if (divisor == 0)
        throw CustomException("Cannot divide because the divisor is 0");

    for (size_t i = 0; i < vec.size(); ++i)
        vec[i] /= divisor;
    return vec;
}

std::vector<double> operator-(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size())
        throw CustomException("Cannot subtract vectors because their lengths are not the same");

    std::vector<double> newVec(lhs.size(), 0);

    for (size_t i = 0; i < lhs.size(); ++i)
        newVec[i] = lhs[i] - rhs[i];
    return newVec;
}

std::vector<double> operator*(const std::vector<std::vector<double>>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() == 0 || lhs[0].size() == 0 || rhs.size() == 0)
        throw CustomException("Cannot take cross product: one of the matrices has no data (0 rows or columns)");
    if (lhs[0].size() != rhs.size())
        throw CustomException("Cannot take cross product: matrices are not of compatible sizes");

    std::vector<double> crossProduct(lhs.size(), 0);
    for (int row = 0; row < lhs.size(); ++row) {
        crossProduct[row] = std::inner_product(lhs[row].begin(), lhs[row].end(), rhs.begin(), 0);
    }

    return crossProduct;
}

std::vector<double> operator*(const int mult, const std::vector<double>& vec) {
    std::vector<double> newVec(vec.size(), 0);
    for (size_t i = 0; i < vec.size(); ++i) {
        newVec[i] = vec[i] * mult;
    }

    return newVec;
}