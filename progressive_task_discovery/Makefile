CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

pilot: src/pilot.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

test: pilot
	python3 -m unittest discover -s tests -v

smoke: pilot
	./pilot --method goal_reuse --k 8 --seed 100 --epsilon .005 --budget 20000 --every 1000

.PHONY: test smoke
