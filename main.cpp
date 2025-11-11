#include <iostream>
#include <conio.h>

using std::cout;
using std::cin;

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}

int main(){
	
	cout << "*****************************************************";
	cout << "==== STARTING NEURAL NETWORK APPLICATION PROGRAM ====";	
	cout << "*****************************************************";

	
	bool running = true;
	int input;
	
	do{
		cin >> input;
		
		switch(input)
		{
			case 1: // XOR LOGIC GATE NEURAL NETWORK
				break;
			case 2: // BINARY TO DECIMAL CONVERTER
				break;
			case 3: // ODD OR EVEN NUMBER CLASSIFIER
				break;
			case 4: // PRIME NUMBER DETECTOR
				break;
			case 5: // LINEAR EQUATION SOLVER
				break;
			case 6: // EXIT PROGRAM
				running = false;
				break;
		}
	}while(running);
	
	
	cout << "*****************************************************";
	cout << "===== ENDING NEURAL NETWORK APPLICATION PROGRAM =====";	
	cout << "*****************************************************";
	
	getch();
	return 0;
}