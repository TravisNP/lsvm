#ifndef OVERLOAD_H
#define OVERLOAD_H

#include <vector>
#include <numeric>
#include <iostream>

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
 * Subtracts two vectors pairwise
 * @param lhs the vector being subtracted to
 * @param rhs the vector that is being subtracted
 * @return the pairwise subtraction
 * @throws CustomException if lengths are not equal
 */
std::vector<double> operator-(const std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * Multiplies two vectors pairwise
 * @param lhs a vector
 * @param rhs a vector of the same length
 * @return the pairwise multiplication
 * @throws CustomException if lengths are not equal
 */
std::vector<double> operator*(const std::vector<double>& lhs, const std::vector<double>& rhs);

/**
 * Multiplies a constant through a vector
 * @param mult the constant
 * @param vec the vector
 * @return the vector with each element multiplied by the constant
 */
std::vector<double> operator*(const int mult, const std::vector<double>& vec);

/**
 * Prints a vector of doubles
 * @param os output stream
 * @param vec the vector
 */
std::ostream& operator<<(std::ostream& os, const std::vector<double>& vec);

#endif