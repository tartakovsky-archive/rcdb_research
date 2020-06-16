all: lib.so

lib.so:
	clang++ -fPIC -shared sb.cc -o lib.so -std=c++17 -O3 -march=native -g -DNDEBUG
