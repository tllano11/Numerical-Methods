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

   //vectors based on CSC format
   vector<double> valuesC;
   vector<int> rows;
   vector<int> pointerBC;
   vector<int> pointerEC;
   vector<int> auxCols;

   //Auxiliar elements to create the "matrix"
   int pos = 0;
   int auxPos = 0;
   pointerBC.push_back(pos);
   for(int i = 0; i < matrixLength; i++){
      if(pos != auxPos){
         pointerBC.push_back(pos);
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
            valuesC.push_back(val);
            rows.push_back(j);
            auxCols.push_back(i); // this variable is used to help to get columns in the CSR format
            matrix[i][j] = val;
         } // if probability is greater than density, do nothing, it doesn't matter
      }
      pointerEC.push_back(pos);
   }
   
   // From CSC format to CSR format.
   // It easier work with CSR format (personal opinion).
   int posi = 0;
   for(int i = 0; i < matrixLength; i++){
      pointerBR.push_back(posi);
      for(int j = 0; j < rows.size(); j++){
         if(rows[j] == i){
            valuesR.push_back(valuesC[j]);
            columns.push_back(auxCols[j]);
            posi++;
         }
      }
      pointerER.push_back(posi);
   }
   cout << "valuesC" << endl;
   for(double n: valuesC){
      cout << n << " ";
   }
   cout << endl;
   cout << "Rows" << endl;
   for(double n: rows){
      cout << n << " ";
   }
   cout << endl;
   cout << "PointerBC" << endl;
   for(double n: pointerBC){
      cout << n << " ";
   }
   cout << endl;
   cout << "PointerEC" << endl;
   for(double n: pointerEC){
      cout << n << " ";
   }
   cout << endl;
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