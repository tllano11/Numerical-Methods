#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <math.h>
#include "jacobi.h"

using namespace std;

double** matrixD;
double** matrixU;
int N = 0;

vector<double> multiplyMatrixVector(double** matrix, vector<double> vectorB){
	vector<double> result;
	double temp;
	for(int i = 0; i < N; i++){
		temp = 0;
		for(int j = 0; j < N; j++){
			temp += matrix[i][j] * vectorB[j];
		}
		result.push_back(temp);
	}
	return result;
}

double** multiplyMatrixMatrix(double** matrix1, double** matrix2){
	double** matrixResult = new double*[N];
   	for(int i = 0; i < N; i++){
    	matrixResult[i] = new double[N];
    }
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			matrixResult[i][j] = 0;
			for(int k = 0; k < N; k++){
				matrixResult[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}
	return matrixResult;
}

void getDandU(double** matrix){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			if(i == j){
				matrixD[i][j] = matrix[i][j];
				matrixU[i][j] = 0;
			}else{
				matrixD[i][j] = 0;
				matrixU[i][j] = -(matrix[i][j]);
			}
		}
	}
}

void printMatrix(double** matrix){
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			cout << setw(10) << matrix[i][j] << " ";
		}
		cout << endl;
	}
}

void printVector(vector<double> vec){
	for(int i = 0; i < N; i++){
		cout << setw(10) << vec[i] << endl;
	}
	cout << endl;
}

void getInverse(){
	for(int i = 0; i < N; i++){
		matrixD[i][i] = pow(matrixD[i][i], -1.0);
	}
}

vector<double> sumVectors(vector<double> vector1, vector<double> vector2){
	vector<double> result;
	for(int i = 0; i < N; i++){
		result.push_back(vector1[i] + vector2[i]);
	}
	return result;
}

double getError(vector<double> vectorX, vector<double> vectorXant){
	double max = 0;
	double tmp = 0;
	int size = vectorX.size();
	for(int i = 0; i < size; i++){
		tmp = abs(vectorX[i] - vectorXant[i]);
		if(tmp > max){
			max = tmp;
		}
	}
	return max;
}

void jacobi(int matrixLength, string matrixFile, string bFile, int maxIterations, double tolerance){

	// Initializing all the variables that will be used in this program
	N = matrixLength;
	double** matrixAux = new double*[N];
	matrixD = new double*[N];
	matrixU = new double*[N];
	double** matrix = new double*[N];
	vector<double> vectorXant;
	vector<double> vectorX;
	vector<double> vectorB;

   	for(int i = 0; i < N; i++){
    	matrix[i] = new double[N];
    	matrixD[i] = new double[N];
    	matrixU[i] = new double[N];
    	matrixAux[i] = new double[N];
    	vectorX.push_back(0);
    }

    // Reading files from file system and put the elements into matrix
    ifstream f(matrixFile);
    for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			f >> matrix[i][j];
		}
    }

    ifstream fin(bFile);
    double x;
    for(int i = 0; i < N; i ++){
    	fin >> x;
    	vectorB.push_back(x);
    }


    double error = tolerance + 1;
	getDandU(matrix);
	getInverse();
	clock_t begin = clock();
	vector<double> vector1 = multiplyMatrixVector(matrixD, vectorB);
	matrixAux = multiplyMatrixMatrix(matrixD, matrixU);

	int cont = 0;
	while(error > tolerance && cont < maxIterations){
		vectorXant = vectorX;
		vector<double> vector2 = multiplyMatrixVector(matrixAux, vectorX);
		vectorX = sumVectors(vector1, vector2);
		error = getError(vectorX, vectorXant);
		cont++;
	}
	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

  	if(error < tolerance){
  		cout << "The solution is: " << endl;
		printVector(vectorX);
		cout <<"It takes " << cont << " iterations" << endl;
  	}else{
  		cout << "Sorry, it failed in " << cont << " iterations" << endl;
  	}
	cout << "time:" << elapsed_secs << endl;
}

void jacobiInit(){
	int matrixLength, maxIterations;
	string matrixFile, bFile;
	double tolerance;
	cout << "Enter matrix length" << endl;
	cin >> matrixLength;
	cout << "Enter matrix A path (filename)" << endl;
	cin >> matrixFile;
	cout << "Enter vector B path (filename)" << endl;
	cin >> bFile;
	cout << "Enter maximum number of iterations" << endl;
	cin >> maxIterations;
	cout << "Enter tolerance" << endl;
	cin >> tolerance;
	jacobi(matrixLength, matrixFile, bFile, maxIterations, tolerance);
}
/*
int main(int argc, char** argv){
	//jacobiInit();
	// Checking number of arguments given by the user
	if(argc != 6){
		cout << "Sorry, wrong arguments" << endl;
		cout << "Usage: ./jacobi matrixLength matrixFile bFile maxIterations tolerance" << endl;
		exit(1);
	}

	jacobi(atoi(argv[1]), argv[2], argv[3], atoi(argv[4]), atof(argv[5]));

}*/
