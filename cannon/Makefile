GCC=mpic++
GCCFLAGS= -I/usr/local/opt/llvm/lib -O3 -march=native -std=c++14 -omp

all: matrix

matrix: cannon.cpp
	$(GCC) $(GCCFLAGS) cannon.cpp -o cannon

clean:
	rm -f cannon
	rm -rf connon.dSYM
