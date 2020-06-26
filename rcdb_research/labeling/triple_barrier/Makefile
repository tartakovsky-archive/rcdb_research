all: lib.so

lib.so:
	clang++ -fPIC -shared tb.cc -o lib.so -std=c++17 -O3 -march=native -g -DNDEBUG
