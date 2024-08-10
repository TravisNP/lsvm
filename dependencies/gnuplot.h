#ifndef GNUPLOT_H
#define GNUPLOT_H

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

/** Plots the decision boundary and the 2D data and colors each point according to their label
 * @param file_name the file to read the data from
 */
void plot2dData(const std::string file_name);

#endif