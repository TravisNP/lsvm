#ifndef DATA_H
#define DATA_H

#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>

/** Saves the data, labels, and decision boundary to a file
 * @param data the data (must have a column of 1s on the left)
 * @param labels the labels (must be -1 and 1)
 * @param decisionBoundary the decision vector
 * @param file_name the file to save the data to
 */
void saveData(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, const std::vector<double>& decisionBoundary, const std::string file_name);

/** loads in data in the format saveData saves it to
 * @param fileName the file to load the data from
 * @param data the data points
 * @param labels the labels
*/
void load2dData(std::string fileName, std::vector<std::vector<double>>& data, std::vector<int>& labels);

/** Populates data and labels with the number of 2d data points specified
 * @param data the data being populated
 * @param labels the labels being populated
 * @param numberSamples the number of samples to be generated
 */
void init2dData(std::vector<std::vector<double>>& data, std::vector<int>& labels, const int numberSamples);

/** Populates data and labels with the number of 3d data points specified
 * @param data the data being populated
 * @param labels the labels being populated
 * @param numberSamples the number of samples to be generated
 */
void init3dData(std::vector<std::vector<double>>& data, std::vector<int>& labels, const int numberSamples);


#endif