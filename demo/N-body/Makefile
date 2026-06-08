CXX := g++

ROOT := ../..

TARGET := simulation
SRC := simulation.cpp
MG_SRC := $(ROOT)/src/multigrid.cpp

CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -fopenmp -I$(ROOT)/include
LDFLAGS := -fopenmp

RESULTS_DIR := results

.PHONY: all run plot gif clean dirs

all: dirs $(TARGET)

dirs:
	mkdir -p $(RESULTS_DIR)

$(TARGET): $(SRC) $(MG_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

run: all
	./$(TARGET)

plot:
	python plot.py

gif:
	python plot.py
	
clean:
	rm -f $(TARGET)
	rm -f $(RESULTS_DIR)/*.csv
	rm -f $(RESULTS_DIR)/*.gif
	rm -f $(RESULTS_DIR)/*.png