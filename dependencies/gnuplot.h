#ifndef GNUPLOT_H
#define GNUPLOT_H

#include <iostream>
#include <vector>
#include <fstream>

/** Plots 2D data and colors each point according to their label
 * @param data the data (must have a column of 1s on the left)
 * @param labels the labels (must be -1 and 1)
 */
void plotData(const std::vector<std::vector<double>>& data, const std::vector<int>& labels);

#endif