all:
	nvcc -c -o main.o main.cu initialization/init.cu solve/solve.cu diagnostics/diagnostics.cu write/write.cpp -I/usr/lib/x86_64-linux-gnu/openmpi/include
	mpic++ -o run_serpentin main.o -L/usr/local/cuda/lib64 -lcudart

clean:
	rm main.o
	rm run_serpentin
