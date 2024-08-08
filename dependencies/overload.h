#ifndef OVERLOAD_H
#define OVERLOAD_H

#include <vector>
#include "custom_exceptions.h"

/**
 * Adds to vectors pairwise
 * @param lhs the vector being added to
 * @param rhs the vector that is being added
 * @return the pairwise sum
 */
std::vector<double>& operator+=(std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * Adds to vectors pairwise
 * @param lhs the vector being subtracted to
 * @param rhs the vector that is being subtracted
 * @return the pairwise subtraction
 */
std::vector<double>& operator-(std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * Multiplies a constant through a vector
 * @param mult the constant
 * @param rhs the vector
 * @return the vector multiplied pairwise be the constant
 */
std::vector<double>& operator*(const int mult, std::vector<double>& rhs);

#endif