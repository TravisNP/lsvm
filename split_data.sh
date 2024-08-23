#!/bin/bash

# Check if the filename is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 filename"
    exit 1
fi

filename=$1

# Calculate total number of lines
total_lines=$(wc -l < "$filename")
lines_60=$((total_lines * 60 / 100))
lines_20=$((total_lines * 20 / 100))

# Split the file into training, testing, and validation datasets
head -n $lines_60 "$filename" > trainingData.dat
tail -n +$((lines_60 + 1)) "$filename" | head -n $lines_20 > testData.dat
tail -n +$((lines_60 + lines_20 + 1)) "$filename" > validationData.dat

