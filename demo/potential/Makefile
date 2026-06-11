CXX := g++

ROOT := ../..

TARGET := get_potential
SRC := get_potential.cpp
MG_SRC := $(ROOT)/src/multigrid.cpp

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -fopenmp -I$(ROOT)/include
LDFLAGS := -fopenmp

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC) $(MG_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

run: all
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f results/*.csv
	rm -f results/*.png
	rm -f results/*.gif