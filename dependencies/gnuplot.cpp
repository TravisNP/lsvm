#include "gnuplot.h"

void plotData(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, const std::vector<double>& decisionBoundary) {
    std::ofstream file("data.dat");
    for (int i = 0; i < data.size(); ++i) {
        file << data[i][1] << " " << data[i][2] << " " << labels[i] << std::endl;
    }
    file.close();

    const double slope = -1 * decisionBoundary[1] / decisionBoundary[2];
    const double yIntercept = -1 * decisionBoundary[0] / decisionBoundary[2];
    // Command to pipe to GNUplot
    std::string cmd = "gnuplot -p -e \"unset colorbox; unset key; set palette defined (-1 'red', 1 'blue'); "
                      "plot 'data.dat' using 1:2:3 with points pointtype 7 pointsize 1 palette, "
                      "x * " + std::to_string(slope) + " + " + std::to_string(yIntercept) + " with lines linetype 1 linecolor 'black'\"";
    system(cmd.c_str());
}