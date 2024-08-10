#include "gnuplot.h"

void plot2dData(std::string file_name) {
    std::vector<double> decisionBoundary;

        // Manually read the first line
    std::ifstream infile("data.dat");
    std::string firstLine;
    if (std::getline(infile, firstLine)) {
        std::istringstream iss(firstLine);
        std::string num;
        for (int i = 0; i < 3; ++i) {
            iss >> num;
            decisionBoundary.push_back(std::stod(num));
        }
    }

    const double slope = -1 * decisionBoundary[1] / decisionBoundary[2];
    const double yIntercept = -1 * decisionBoundary[0] / decisionBoundary[2];
    // Command to pipe to GNUplot
    std::string cmd = "gnuplot -p -e \"unset colorbox; unset key; set palette defined (-1 'red', 1 'blue'); "
                      "plot 'data.dat' using 1:2:3 skip 1 with points pointtype 7 pointsize 1 palette, "
                      "x * " + std::to_string(slope) + " + " + std::to_string(yIntercept) + " with lines linetype 1 linecolor 'black'\"";
    system(cmd.c_str());
}