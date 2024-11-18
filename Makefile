all:
	nvcc main.cu write/write.cpp initialization/init.cu solve/solve.cu diagnostics/diagnostics.cu -o run_serpentin

clean:
	rm run_serpentin
