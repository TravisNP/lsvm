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
        throw CustomException("Cannot add vectors because their lengths are not the same");

    for (size_t i = 0; i < lhs.size(); ++i)
        lhs[i] -= rhs[i];
    return lhs;
}

std::vector<double>& operator*(const int mult, std::vector<double>& rhs) {
    for (size_t i = 0; i < rhs.size(); ++i) {
        rhs[i] *= mult;
    }

    return rhs;
}