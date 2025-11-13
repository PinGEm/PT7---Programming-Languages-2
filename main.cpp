#include <iostream>
#include <conio.h>
#include <cmath>
#include <ctime>
#include <limits>

#define XOR_LOGICGATE_NN 1
#define BINARY_DECIMAL 2
#define ODD_OR_EVEN_CLASSIFIER 3
#define LINEAR_EQUATION_SOLVER 4


using std::cout;
using std::cin;

class NeuralNetwork {
	private:
		// Activation function mapping inputs between values of 0-1
		static double sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}
		
		// Backpropogation function used for readjusting the weights of the training data
		static double sigmoidDerivative(double x) {
			return x * (1.0 - x);
		}

		double learningRate = 0.5; // rate at which the AI learns

	public:
	
		// Represents each individual neuron inside the neural network
		class Neuron {
			public:
				// Initiate variables
				double* weights;
				int num_weights;
				double bias;
				double output;
				double delta; // not used within this class, but used when training data.
				
				// When this class is instantiated, do the following:
				Neuron(int num_inputs) : num_weights(num_inputs) {
					// for now, randomly set the weights and biases of this neuron.
					weights = new double[num_weights];
					for (int i = 0; i < num_weights; ++i) {
						weights[i] = ((double)rand() / RAND_MAX) - 0.5;
					}
					bias = ((double)rand() / RAND_MAX) - 0.5;
				}

				~Neuron() {
					// Free up memory
					delete[] weights;
				}

				double activate(double* inputs) {
					// consider this the output of this neuron.
					double sum = bias;
					for (int i = 0; i < num_weights; ++i) {
						sum += inputs[i] * weights[i];
					}
					output = NeuralNetwork::sigmoid(sum); // evaluate the output between the inputs based on the weights and bias
					return output;
				}
		};
		
		// Represents each layer inside the neural network
		class Layer {
			public:
				// Initiate variables
				Neuron** neurons; // create a dynamic array of neurons
				int num_neurons;
				
				// Create all the necessary neurons for this layer.
				Layer(int num_neurons_, int num_inputs_per_neuron) : num_neurons(num_neurons_) {
					neurons = new Neuron * [num_neurons];
					for (int i = 0; i < num_neurons; ++i) {
						// Each neuron is created with the proper inputs per neuron
						neurons[i] = new Neuron(num_inputs_per_neuron);
					}
				}

				~Layer() {
					// free up all the neurons. make sure to delete all the neurons in the array.
					for (int i = 0; i < num_neurons; ++i) {
						delete neurons[i];
					}
					delete[] neurons;
				}
				
				// outputting all the neurons 
				double* feedForward(double* inputs) {
					double* outputs = new double[num_neurons];
					for (int i = 0; i < num_neurons; ++i) {
						// iterate through each neuron inside this layer and call the sigmoid function inside them.
						outputs[i] = neurons[i]->activate(inputs);
					}
					
					// return all the outputs.
					return outputs; // note: make sure to call the delete[] when using this!
				}
		};
		
		// Creation of variables for the layers of the neural network
		Layer** layers;
		int num_layers;

		// Instantiate all the layers and the number of neurons inside each layer.
		NeuralNetwork(int* topology, int size) : num_layers(size - 1) {
			layers = new Layer * [num_layers];
			for (int i = 0; i < num_layers; ++i) {
				// create all the layers and the necessary neurons and input for it
				layers[i] = new Layer(topology[i + 1], topology[i]);
			}
		}

		// Free up the memory and delete all the layers.
		~NeuralNetwork() {
			for (int i = 0; i < num_layers; ++i) {
				delete layers[i];
			}
			delete[] layers;
		}
		
		// Based on the neural network, predict and evaluate the right output value based on the input.
		double* predict(double* inputs, int input_size) {
			
			// Create the array of inputs and their correct size
			double* current_inputs = new double[input_size];
			for (int i = 0; i < input_size; ++i)
			{
				current_inputs[i] = inputs[i];
			}
			
			// Iterate through each layer and feed them the input.
			for (int l = 0; l < num_layers; ++l) {
				// get the output from feeding the inputs to the layers, then set the current input the output value.
				double* outputs = layers[l]->feedForward(current_inputs);
				if (l > 0) 
				{
					delete[] current_inputs;
				}
				current_inputs = outputs;
			}
			
			// Return the predicted value
			return current_inputs; // make sure to handle deletion of the output here !
		}
		
		// Adjust for the neural network's ACTUAL weights and biases based on the expected value and trained data.
		void train(double* inputs, double* expected, int input_size) {
			double* outputs = predict(inputs, input_size); // to get the initial evaluation of the inputs

			// Compute delta/error for each neuron in the output layer
			Layer* outputLayer = layers[num_layers - 1];
			for (int i = 0; i < outputLayer->num_neurons; ++i) {
				// access each neuron's delta and compute the necessary adjustments based on the output, expected and the sigmoid
				double out = outputLayer->neurons[i]->output;
				outputLayer->neurons[i]->delta = (expected[i] - out) * sigmoidDerivative(out);
			}

			// Compute delta/error for each neuron in the hidden layers
			for (int l = num_layers - 2; l >= 0; --l) {
				Layer* current = layers[l];
				Layer* next = layers[l + 1];
				for (int i = 0; i < current->num_neurons; ++i) {
					// access each neuron's delta and compute the necessary adjustments based on errors of the NEXT layers
					double error = 0.0;
					for (int j = 0; j < next->num_neurons; ++j) {
						// we also gradually compute the adjustments for the next layer.
						error += next->neurons[j]->weights[i] * next->neurons[j]->delta;
					}
					current->neurons[i]->delta = error * sigmoidDerivative(current->neurons[i]->output);
				}
			}

			// Update the weights and biases in each layers neurons
			for (int l = 0; l < num_layers; ++l) {
				Layer* layer = layers[l];
				double* layerInputs;
				int num_inputs;
				
				
				if (l == 0) {
					// IF WE ARE AT THE INPUT LAYER
					layerInputs = inputs;
					num_inputs = input_size;
				}
				else {
					// IF WE ARE AT THE HIDDEN/OUTPUT LAYERS
					Layer* prevLayer = layers[l - 1];
					num_inputs = prevLayer->num_neurons;
					layerInputs = new double[num_inputs];
					for (int i = 0; i < num_inputs; ++i)
					{
						// store the outputs of each neuron inside the layerInputs.
						layerInputs[i] = prevLayer->neurons[i]->output;
					}
				}
				
				// 
				for (int n = 0; n < layer->num_neurons; ++n) {
					// adjust the weights and bias of each neuron now that we have computed the delta.
					Neuron* neuron = layer->neurons[n];
					for (int w = 0; w < neuron->num_weights; ++w) 
					{
						// compute the weights based on learning rate, delta, and the layers output
						neuron->weights[w] += learningRate * neuron->delta * layerInputs[w];
					}
					neuron->bias += learningRate * neuron->delta;
				}

				if (l != 0) 
				{
					// if this is the input layer, just delete this.
					delete[] layerInputs;
				}
			}
		
		// clear up the memory now that we don't need to store this data
		delete[] outputs;
	}
};

int validateInput(int low, int high);

int main() {
	srand(time(NULL)); // for the initial random generation of the weights for neural network
    cout << "*****************************************************\n";
    cout << "==== STARTING NEURAL NETWORK APPLICATION PROGRAM ====\n";
    cout << "*****************************************************\n";

    bool running = true;
    int input;

	// Initiate Variables for the Neural Network Program
    int *topology = nullptr; // to setup multiple different layers
    NeuralNetwork *nn = nullptr; // neural network itself
	
	double input_data_one;
	double input_data_two;
	double input_data_three; // USED IN BINARY-DECIMAL CONVERTER & LINEAR EQUATION SOLVER
	double input_data_four; // USED IN BINARY-DECIMAL-CONVERTER
	double* output; // output of the user's input based on the training data
	
	
	// ============================ TRAINING DATA START ============================
	
	// --------------- XOR SPECIFIC
	double training_inputs_XOR[4][2] = 
	{
		{0,0}, {0,1}, {1,0}, {1,1}
	};
	double expected_outputs_XOR[4][1] = 
	{
		{0}, {1}, {1}, {0}
	};
	double input_set_XOR[1][2];
	
	
	// --------------- BINARY-DECIMAL SPECIFIC
	double training_inputs_BD[16][4] = {
		{0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1},
		{0,1,0,0}, {0,1,0,1}, {0,1,1,0}, {0,1,1,1},
		{1,0,0,0}, {1,0,0,1}, {1,0,1,0}, {1,0,1,1},
		{1,1,0,0}, {1,1,0,1}, {1,1,1,0}, {1,1,1,1}
	};
	
	double expected_outputs_BD[16][1] = {
		{0.0/15}, {1.0/15}, {2.0/15}, {3.0/15},
		{4.0/15}, {5.0/15}, {6.0/15}, {7.0/15},
		{8.0/15}, {9.0/15}, {10.0/15}, {11.0/15},
		{12.0/15}, {13.0/15}, {14.0/15}, {15.0/15}
	};
	double input_set_BD[1][4];
	
	
	// --------------- ODD-OR-EVEN SPECIFIC (turned into binary)
	double training_inputs_OE[50][17] = {};
	
	for (int i = 0; i < 50; ++i) {
		int num = i;
        for (int b = 0; b < 17; ++b) {
            // Fill bits MSB first
            training_inputs_OE[i][17 - 1 - b] = (num >> b) & 1;
        }
	}

	double expected_outputs_OE[50][1] = {
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1},
		{0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}, {0}, {1}
	};

	double input_set_OE[1][17];
	int tempNum;
	
	
	// --------------- Linear Equation Solver (ax + b = c) : NORMALIZED VALUES OF THE NUMBERS AS STATED
	double training_inputs_LES[15][3] = {
		{0.1111, 0.5000, 0.6000}, // a=1, b=0, c=2  -> x=(2-0)/1=2
		{0.2222, 0.4500, 0.5500}, // a=2, b=-1, c=1 -> x=1
		{0.3333, 0.6000, 0.7000}, // a=3, b=2, c=4  -> x=0.667
		{0.4444, 0.4000, 0.5000}, // a=4, b=-2, c=0 -> x=0.5
		{0.5556, 0.5500, 0.6500}, // a=5, b=1, c=3  -> x=0.4
		{0.6667, 0.3000, 0.5000}, // a=6, b=-4, c=0 -> x=0.667
		{0.7778, 0.5000, 0.7000}, // a=7, b=0, c=4  -> x=0.571
		{0.8889, 0.6000, 0.8000}, // a=8, b=2, c=6  -> x=0.5
		{1.0000, 0.5000, 0.6000}, // a=9, b=0, c=2  -> x=0.222
		{0.3333, 0.7000, 0.5000}, // a=3, b=4, c=0  -> x=-1.333
		{0.4444, 0.4000, 0.7000}, // a=4, b=-2, c=4 -> x=1.5
		{0.5556, 0.4500, 0.5500}, // a=5, b=-1, c=1 -> x=0.4
		{0.6667, 0.6500, 0.4500}, // a=6, b=3, c=-1 -> x=-0.667
		{0.7778, 0.3500, 0.6500}, // a=7, b=-3, c=3 -> x=0.857
		{0.8889, 0.5500, 0.4500}, // a=8, b=1, c=-1 -> x=-0.25
	};

	double expected_outputs_LES[15][1] = {
		{0.6},   // x=2   → (2+10)/20
		{0.55},  // x=1
		{0.533}, // x≈0.667
		{0.525}, // x=0.5
		{0.52},  // x=0.4
		{0.533}, // x≈0.667
		{0.5286},// x≈0.571
		{0.525}, // x=0.5
		{0.511}, // x≈0.222
		{0.433}, // x≈-1.333
		{0.575}, // x=1.5
		{0.52},  // x=0.4
		{0.4665},// x=-0.667
		{0.5428},// x≈0.857
		{0.4875},// x=-0.25
	};
	double input_set_LES[1][3];
	
	// ============================ TRAINING DATA END ============================


	
	// ------------------ MAIN APPLICATION LOOP ------------------ //
    do {
		
		cout << "WHAT NEURAL NETWORK APPLICATION WOULD YOU LIKE TO USE:\n";
		cout << "Type '1' for XOR Logic Gate Neural Network\n";
		cout << "Type '2' for Binary -> Decimal Converter\n";
		cout << "Type '3' for Odd or Even Number Classifier\n";
		cout << "Type '4' for Linear Equation Solver\n";
		cout << "Type '5' to Exit Program\n";
		
        cout << "Input: ";
        cin >> input;
		cout << '\n';
		
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        switch (input) {
        case XOR_LOGICGATE_NN:
			// Setup Variables: Layers
			topology = new int[3];
			topology[0] = 2; topology[1] = 2; topology[2] = 1;
			nn = new NeuralNetwork(topology, 3);
			
			// Adjusting weights to be accurate
			for (int epoch = 0; epoch < 10000; ++epoch) {
				for (int i = 0; i < 4; ++i)
				{
					nn->train(training_inputs_XOR[i], expected_outputs_XOR[i], 2);	
				}
			}
			
			// Asking for User Input
			cout << "FIRST VALUE: \n";
			input_data_one = round(validateInput(0,1));
			cout << "SECOND VALUE: \n";
			input_data_two = round(validateInput(0,1));
			
			// input user's input into the input set
			input_set_XOR[0][0] = input_data_one;
			input_set_XOR[0][1] = input_data_two;
			
			// Predict Results
			output = nn->predict(input_set_XOR[0],2);
			cout << "XOR Result: " << output[0] << "\n";
			(output[0] > 0.5) ? cout << "Value: TRUE\n" : cout << "Value: FALSE\n";
            break;
        case BINARY_DECIMAL: // BINARY-DECIMAL-CONVERTER
			// Setup Variables: Layers
			topology = new int[3];
			topology[0] = 4; topology[1] = 8; topology[2] = 1;
			nn = new NeuralNetwork(topology,3);
			
			// Adjusting weights to be accurate
			for(int epoch = 0; epoch < 30000; ++epoch){
				for(int i = 0; i < 17; ++i)
				{
					nn->train(training_inputs_BD[i], expected_outputs_BD[i], 4);
				}
			}
			
			// Asking for User Input
			cout << "FIRST VALUE: \n";
			input_data_one = round(validateInput(0,1));
			cout << "SECOND VALUE: \n";
			input_data_two = round(validateInput(0,1));
			cout << "THIRD VALUE: \n";
			input_data_three = round(validateInput(0,1));
			cout << "FOURTH VALUE: \n";
			input_data_four = round(validateInput(0,1));
			
			// input user's input into the input set
			input_set_BD[0][0] = input_data_one;
			input_set_BD[0][1] = input_data_two;
			input_set_BD[0][2] = input_data_three;
			input_set_BD[0][3] = input_data_four;
			
			// Predict Results
			output = nn->predict(input_set_BD[0],4);
			cout << "BINARY NUMBER: " << input_data_one << input_data_two << input_data_three << input_data_four << '\n';
			cout << "Result: " << output[0] << "\n";
			cout << "Decimal Result: " << (int)round((output[0] * 15)) << "\n";
            break;
        case ODD_OR_EVEN_CLASSIFIER:
			// Setup Variables: Layers
			topology = new int[3];
			topology[0] = 17; topology[1] = 8; topology[2] = 1;
			nn = new NeuralNetwork(topology,3);
						
			// Adjusting weights to be accurate
			for(int epoch = 0; epoch < 10000; ++epoch){
				for(int i = 0; i < 50; ++i)
				{
					nn->train(training_inputs_OE[i], expected_outputs_OE[i], 17);
				}
			}
			
			// Asking for User Input
			cout << "NUMBER (0-100,000): \n";
			input_data_one = abs(validateInput(-100000,100000));
			input_set_OE[0][0] = input_data_one;
			
			
			// Turning number into a binary
			tempNum = input_data_one;
			for (int b = 0; b < 17; ++b) {
				input_set_OE[0][17 - 1 - b] = (tempNum >> b) & 1;
			}
			
			// Predict Results
			output = nn->predict(input_set_OE[0], 17);
			cout << "Result: " << output[0] << "\n";
			cout << "Binary (INPUT DATA): ";
			
			// Displaying the full 17 bit binary
			for(int i = 0; i < 17; i++)
			{
				cout << input_set_OE[0][i];
			}
			
			cout << "\n";
			
			(output[0] > 0.5) ? cout << "Number is ODD" : cout << "Number is EVEN";
            break;
        case LINEAR_EQUATION_SOLVER:
			cout << "Linear Equation Solver: ax + b = c\n";
			// Setup Variables: Layers
			topology = new int[3];
			topology[0] = 3; topology[1] = 10; topology[2] = 1;
			nn = new NeuralNetwork(topology,3);
			
			// Adjusting weights to be accurate
			for(int epoch = 0; epoch < 25000; ++epoch){
				for(int i = 0; i < 15; ++i)
				{
					nn->train(training_inputs_LES[i], expected_outputs_LES[i], 3);
				}
			}
			
			// Asking for User Input
			cout << "FIRST VALUE: \n";
			input_data_one = validateInput(0, 9);
			cout << "SECOND VALUE: \n";
			input_data_two = validateInput(-10, 10);
			cout << "THIRD VALUE: \n";
			input_data_three = validateInput(-10, 10);
			
			// Normalize the values before sending it in the input set (Making it so that the value is between 0-1.0)
			input_set_LES[0][0] = input_data_one / 9.0;
			input_set_LES[0][1] = (input_data_two + 10.0) / 20.0;
			input_set_LES[0][2] = (input_data_three + 10.0) / 20.0;
			
			// Predict Results (and denormalizing of the value)
			cout << "---------\n";
			output = nn->predict(input_set_LES[0], 3);
			cout << "Result = " << output[0] << "\n";
			cout << "Denormalized Result (x) = " << output[0] * 20.0 - 10.0 << '\n';
			cout << "Formula: " << input_data_one << '(' << output[0] * 20.0 - 10.0 << ") + " << input_data_two << " = " << input_data_three << '\n';
			cout << "Expected X = " << (input_data_three - input_data_two) / input_data_one << '\n';
			cout << "Absolute Error = " << fabs(output[0] * 20.0 - 10.0 - (input_data_three - input_data_two) / input_data_one) << '\n'; // absolute value of the difference between predicted x and expected x
			cout << "Margin of Error = " << input_data_three - (input_data_one * (output[0] * 20.0 - 10.0) + input_data_two) << '\n';
            break;
        case 5:
            running = false;
            break;
        };
		
		// Cleanup Memory for the Next Neural Network Application
		delete nn;
		delete[] topology;
		delete[] output;
		output = nullptr;
		topology = nullptr;
		nn = nullptr;
		
		(input != 5) ? cout << "\n**************************\n" : cout << "\n";
    } while (running);

    cout << "*****************************************************\n";
    cout << "===== ENDING NEURAL NETWORK APPLICATION PROGRAM =====\n";
    cout << "*****************************************************\n";

    getch();
    return 0;
}

int validateInput(int low, int high){
	// check if users input is between the lowest value and highest, clear for any errors in the input
	int input = low - 1;
	cout << "Enter a value from " << low << "-" << high << ": ";
	
	do{
		cin >> input;
		std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		(input < low || input > high) ? cout << "Error! Please input a number from " << low << "-" << high << ": " : cout << "";
	}while(input < low || input > high);
		
	return input;
}