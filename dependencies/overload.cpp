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

std::vector<double> operator*(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size())
        throw CustomException("Cannot multiply vectors because their lengths are not the same");

    std::vector<double> newVec(lhs.size(), 0);

    for (size_t i = 0; i < lhs.size(); ++i)
        newVec[i] = lhs[i] * rhs[i];
    return newVec;
}

std::vector<double> operator*(const int mult, const std::vector<double>& vec) {
    std::vector<double> newVec(vec.size(), 0);
    for (size_t i = 0; i < vec.size(); ++i) {
        newVec[i] = vec[i] * mult;
    }

    return newVec;
}

std::vector<double> operator*(const double mult, const std::vector<double>& vec) {
    std::vector<double> newVec(vec.size(), 0);
    for (size_t i = 0; i < vec.size(); ++i) {
        newVec[i] = vec[i] * mult;
    }

    return newVec;
}