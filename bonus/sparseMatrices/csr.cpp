#include <stdio.h>
#include <random>
#include <iostream>
#include <vector>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <map>
#include <algorithm> 
#include <string>
#include <unistd.h>
#include <time.h>
#include <iomanip>

using namespace std;
int matrixLength;
double matrixDensity;

// vectors based on CSR format
vector<double> valuesR;
vector<int> columns;
vector<int> pointerBR;
vector<int> pointerER;

void printMatrix(double **matrix){
   for(int i = 0; i < matrixLength; i++){
      for(int j = 0; j < matrixLength; j++){
         cout << setw(10) << matrix[i][j] << " | ";
      }
      cout << endl;
   }
}

void printVector(vector<double> vec){
   for(int i = 0; i < matrixLength; i++){
      cout << setw(10) << vec[i] << endl;
   }
   cout << endl;
}

vector<double> multiply(){
   vector<double> auxR;
   vector<double> r;
   for(int i = 0; i < matrixLength; i++){
      double val = rand() % 10 + 1;
      auxR.push_back(0.0);
      r.push_back(val);
   }
   cout << "r" << endl;
   printVector(r);
   for (int i = 0; i < matrixLength; i++) {
      for (int j = pointerBR[i]; j < pointerER[i]; j++) {
         auxR[i] += (valuesR[j] * r[i]);
      }
   }
   return auxR;
}

void createFormatMatrix(){
   srand(time(NULL));

   double val;

   double** matrix = new double*[matrixLength];

   for(int i = 0; i < matrixLength; i++){
      matrix[i] = new double[matrixLength];
   }

   //Auxiliar elements to create the "matrix"
   int pos = 0;
   int auxPos = 0;
   pointerBR.push_back(pos);
   for(int i = 0; i < matrixLength; i++){
      if(pos != auxPos){
         pointerBR.push_back(pos);
      }
      auxPos = pos;
      // Generate CSC format
      for(int j = 0; j < matrixLength; j++){
         // probability variable is used to check the density of the matrix based on probabilities
         double probability = (double) rand() / (RAND_MAX);
         // if probabilidy is less than the density we can add a value different from zero.
         if(probability < matrixDensity){
            pos++;
            val = rand() % 10 + 1;// (RAND_MAX); // a random value to add to values vector of CSC format
            valuesR.push_back(val);
            columns.push_back(j);
            matrix[i][j] = val;
         } // if probability is greater than density, do nothing, it doesn't matter
      }
      pointerER.push_back(pos);
   }
   
   cout << "valuesR" << endl;
   for(double n: valuesR){
      cout << n << " ";
   }
   cout << endl;
   cout << "Columns" << endl;
   for(double n: columns){
      cout << n << " ";
   }
   cout << endl;
   cout << "pointerBR" << endl;
   for(double n: pointerBR){
      cout << n << " ";
   }
   cout << endl;
   cout << "PointerER" << endl;
   for(double n: pointerER){
      cout << n << " ";
   }
   cout << endl;
   cout << "Matrix" << endl;
   printMatrix(matrix);
}

int main(int argc, char** argv) {
   if(argc != 4){
      cerr << "Error: Invalid arguments. " << endl;
      exit(1);
   }
   
   //Get the params given by the user.
   matrixLength = atoi(argv[1]); // the length of the matrix
   matrixDensity = atof(argv[2]); // portion of the matrix that will be different from zero (between 0 and 1)

   createFormatMatrix();
   vector<double> result = multiply();
   cout << "Result vector" << endl;
   printVector(result);

}