CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
branch: src/branch.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
test: branch
	python3 -m unittest discover -s tests -v
.PHONY: test
