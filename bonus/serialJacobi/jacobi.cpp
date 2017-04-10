#include <iostream>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <iomanip>
#include <ctime>
#include <fstream>

#define N 1000
using namespace std;
double** matrixD = new double*[N];
double** matrixU = new double*[N];
vector<double> vectorB;
vector<double> vectorX;
double** matrix = new double*[N];

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

void generate(){
	double tmp = 0.1;
	for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			if(i != j){
				matrix[i][j] = 0.1 + tmp;//(rand() % 100 + 1)/matrix[i][i];
			}else{
				matrix[i][j] = 500000 + tmp;//(rand() % 100 + 1);
			}
			tmp += 0.1;
		}
		vectorX.push_back(0);
		vectorB.push_back(rand() % 10 + 1);
	}
}

int main(int argc, char** argv){
	double** matrixAux = new double*[N];

   	for(int i = 0; i < N; i++){
    	matrix[i] = new double[N];
    	matrixD[i] = new double[N];
    	matrixU[i] = new double[N];
    	matrixAux[i] = new double[N];
    	vectorX.push_back(0);
    }

	/*for(int i = 0; i < N; i++){
		for(int j = 0; j < N; j++){
			matrix[i][j] = rand() % 10 + 1;
		}
		vectorX.push_back(0);
		vectorB.push_back(rand() % 10 + 1);
	}*/



    ifstream f("../../a.csv");
    for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			f >> matrix[i][j];
		}
    }
    ifstream fin("../../b.csv");
    double x;
    for(int i = 0; i < N; i ++){
    	fin >> x;
    	vectorB.push_back(x);
    }
    //cout << "initial matrtix" << endl;
    //printMatrix(matrix);
	getDandU(matrix);
	getInverse();
	vector<double> sol;
	clock_t begin = clock();
	vector<double> vector1 = multiplyMatrixVector(matrixD, vectorB);
	matrixAux = multiplyMatrixMatrix(matrixD, matrixU);
	for(int i = 0; i < 100; i++){
		vector<double> vector2 = multiplyMatrixVector(matrixAux, vectorX);
		vectorX = sumVectors(vector1, vector2);
		//printVector(vectorX);
	}
	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;


	/*cout << "A" << endl;
	printMatrix(matrixD);

	cout << "U" << endl;
	printMatrix(matrixU);

	cout << "VectorB" << endl;
	printVector(vectorB);*/

	cout << "VectorX" << endl;
	printVector(vectorX);
	cout << "time:" << elapsed_secs << endl;
}
