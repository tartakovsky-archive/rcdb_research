all: lib.so

lib.so:
	clang++ -shared -fPIC -o lib.so lib.cc -std=c++17 -O3 -ffast-math -lomp -march=native -Xpreprocessor -fopenmp -lpthread -Wall -Wextra

clean:
	rm lib.so


lib_sanitize.so:
	clang++ -shared -fPIC -o lib_sanitize.so lib.cc -std=c++17 -O3 -ffast-math -lomp -march=native -Xpreprocessor -fopenmp -lpthread -g -fsanitize=undefined -fsanitize=address -fsanitize=leak 

sanitize: lib_sanitize.so
	clang++ -o sanitize sanitize.cc -std=c++17 -O3 -ffast-math -lomp -march=native -Xpreprocessor -fopenmp -lpthread -g ./lib_sanitize.so -fsanitize=undefined -fsanitize=address -fsanitize=leak 

clean_sanitize:
	rm lib_sanitize.so
	rm sanitize