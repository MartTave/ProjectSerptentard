all:
	g++ main.cpp write/write.cpp initialization/init.cpp solve/solve.cpp diagnostics/diagnostics.cpp -o run_serpentin

clean:
	rm run_serpentin