nonLinealEquations = ./nonLinealEquations/nonLinealEquations
jacobi = ./serialJacobi/jacobi
csr = ./sparseMatrices/csr
all: implementations

implementations: nonLinealEquations.o jacobi.o csr.o Main.cpp
	g++ -I./nonLinealEquations -I./serialJacobi -I./sparseMatrices -o $@ $^

nonLinealEquations.o: $(nonLinealEquations).cpp $(nonLinealEquations).h
	g++ -I./nonLinealEquations -c $(nonLinealEquations).cpp

jacobi.o: $(jacobi).cpp $(jacobi).h
	g++ -I./serialJacobi -c $(jacobi).cpp

csr.o: $(csr).cpp $(csr).h
	g++ -I./sparseMatrices -c $(csr).cpp

clean:
	rm -f implementations
	rm -f *.o
