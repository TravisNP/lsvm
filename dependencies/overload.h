#ifndef OVERLOAD_H
#define OVERLOAD_H

#include <vector>
#include <numeric>

#include "custom_exceptions.h"

/**
 * Adds two vectors pairwise
 * @param lhs the vector being added to
 * @param rhs the vector that is being added
 * @return the pairwise sum
 * @throws CustomException if lengths are not equal
 */
std::vector<double>& operator+=(std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * Subtracts two vectors pairwise, editing the first
 * @param lhs the vector being subtracted to
 * @param rhs the vector that is being subtracted
 * @return the pairwise subtraction
 * @throws CustomException if lengths are not equal
 */
std::vector<double>& operator-=(std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * Divides a constant through a vector
 * @param divisor the number being divided by
 * @param vec the vector
 * @return the vector with each element divided by the constant
 * @throws CustomException if divisor is 0
 */
std::vector<double>& operator/=(std::vector<double>& vec, const int divisor);

/**
 * Subtracts to vectors pairwise
 * @param lhs the vector being subtracted to
 * @param rhs the vector that is being subtracted
 * @return the pairwise subtraction
 * @throws CustomException if lengths are not equal
 */
std::vector<double> operator-(const std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * The cross product of two matrices
 * @param lhs the left matrix
 * @param rhs the right matrix (must be n x 1)
 * @return the cross product
 * @throws CustomException if one of the matrices has 0 dimension or if the dimensions needed do not match (m x n * n x 1)
 */
std::vector<double> operator*(const std::vector<std::vector<double>>& lhs, const std::vector<double>& rhs);

/**
 * Multiplies a constant through a vector
 * @param mult the constant
 * @param vec the vector
 * @return the vector with each element multiplied by the constant
 */
std::vector<double> operator*(const int mult, const std::vector<double>& vec);

#endif