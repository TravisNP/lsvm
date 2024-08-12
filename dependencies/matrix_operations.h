#ifndef MATRIX_OPERATIONS_H
#define MATRIX_OPERATIONS_H

#include <vector>

#include "custom_exceptions.h"

/**
 * The inner product of two vectors
 * @param lhs the left matrix
 * @param rhs the right matrix
 * @return the cross product
 * @throws CustomException if one of the matrices has 0 dimension or if the dimensions needed do not match (m x n * n x 1)
 */
double inner_product(const std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * The cross product of two matrices
 * @param lhs the left matrix
 * @param rhs the right matrix (must be n x 1)
 * @return the cross product
 * @throws CustomException if one of the matrices has 0 dimension or if the dimensions needed do not match (m x n * n x 1)
 */
std::vector<double> cross_product(const std::vector<std::vector<double>>& lhs, const std::vector<double>& rhs);

#endif