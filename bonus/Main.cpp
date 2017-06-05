#include <iostream>
#include <math.h>
#include <iomanip>
#include <cmath>
#include "nonLinealEquations.h"
#include "jacobi.h"
#include "csr.h"

using namespace std;

int main(){
	while(cin){
		int methodNumber;
		cout << "Solve non lineal equations. Choose a method: \n"
			  << "\tEquation: e^(-x) -x \n"
			  << "\t1 ---- Incremental searches \n"
			  << "\t2 ---- Bisection \n"
			  << "\t3 ---- False Rule \n"
			  << "\t4 ---- Fixed Point \n"
			  << "\t5 ---- Newton \n"
			  << "\t6 ---- Secant \n"
			  << "\t7 ---- Multiple roots \n"
			  << "Solve system of linear equations. Choose a method:\n"
			  << "\t8 ---- Jacobi\n"
			  << "Types of matrices:\n"
			  << "\t9 ---- create CSR format (sparse matrices)\n"
			  << "\n"
			  << "\t10 ---- Exit\n"
			  << "> ";
		cin >> methodNumber;
		switch(methodNumber){
			case 1:
				incrementalSearches();
				break;
			case 2:
				bisection();
				break;
			case 3:
				falseRule();
				break;
			case 4:
				fixedPoint();
				break;
			case 5:
				newton();
				break;
			case 6:
				secant();
				break;
			case 7:
				multipleRoots();
				break;
			case 8:
				jacobiInit();
				break;
			case 9:
				csrInit();
				break;
			case 10:
				return 0;
			default:
				cout << "Sorry, wrong number" << endl;
		}	
	}
	return 0;
}