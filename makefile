# use g++ compiler
CXX = g++

# C++ version 23, -g is for debugging, -Wall and -Wextra are for compile warning messages
CXXFLAGS = -std=c++23 -g -Wall -Wextra

# Executable name
TARGET = testLSVM

# Source files
SRCS = LSVM.cpp testLSVM.cpp

# Create object files
OBJS = $(SRCS:.cpp=.o)

# Link object file and compile executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Remove object files and executable
clean:
	rm -f $(TARGET) $(OBJS)
