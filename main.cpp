#include <iostream>
#include <conio.h>
#include <cmath>
#include <ctime>
#include <limits>

using std::cout;
using std::cin;

class NeuralNetwork {
	private:
		double sigmoid(double x) {
			return 1 / (1 + exp(-x));
		}

		double sigmoidDerivative(double x) {
			return x * (1.0 - x);
		}

		double learningRate = 0.5;

	public:
		class Neuron {
			public:
				double* weights;
				int num_weights;
				double bias;
				double output;
				double delta;

				Neuron(int num_inputs) : num_weights(num_inputs) {
					weights = new double[num_weights];
					for (int i = 0; i < num_weights; ++i) {
						weights[i] = ((double)rand() / RAND_MAX) - 0.5;
					}
					bias = ((double)rand() / RAND_MAX) - 0.5;
				}

				~Neuron() {
					delete[] weights;
				}

				double activate(double* inputs) {
					double sum = bias;
					for (int i = 0; i < num_weights; ++i) {
						sum += inputs[i] * weights[i];
					}
					output = 1 / (1 + exp(-sum)); // sigmoid
					return output;
				}
		};

		class Layer {
			public:
				Neuron** neurons;
				int num_neurons;

				Layer(int num_neurons_, int num_inputs_per_neuron) : num_neurons(num_neurons_) {
					neurons = new Neuron * [num_neurons];
					for (int i = 0; i < num_neurons; ++i) {
						neurons[i] = new Neuron(num_inputs_per_neuron);
					}
				}

				~Layer() {
					for (int i = 0; i < num_neurons; ++i) {
						delete neurons[i];
					}
					delete[] neurons;
				}

				double* feedForward(double* inputs) {
					double* outputs = new double[num_neurons];
					for (int i = 0; i < num_neurons; ++i) {
						outputs[i] = neurons[i]->activate(inputs);
					}
					return outputs;
				}
		};

		Layer** layers;
		int num_layers;

		NeuralNetwork(int* topology, int size) : num_layers(size - 1) {
			layers = new Layer * [num_layers];
			for (int i = 0; i < num_layers; ++i) {
				layers[i] = new Layer(topology[i + 1], topology[i]);
			}
			srand(time(0));
		}

		~NeuralNetwork() {
			for (int i = 0; i < num_layers; ++i) {
				delete layers[i];
			}
			delete[] layers;
		}

		double* predict(double* inputs, int input_size) {
			double* current_inputs = new double[input_size];
			for (int i = 0; i < input_size; ++i) current_inputs[i] = inputs[i];

			for (int l = 0; l < num_layers; ++l) {
				double* outputs = layers[l]->feedForward(current_inputs);
				if (l > 0) delete[] current_inputs;
				current_inputs = outputs;
			}
			return current_inputs; // caller should delete[]
		}

		void train(double* inputs, double* expected, int input_size) {
			double* outputs = predict(inputs, input_size);

			// Compute delta for output layer
			Layer* outputLayer = layers[num_layers - 1];
			for (int i = 0; i < outputLayer->num_neurons; ++i) {
				double out = outputLayer->neurons[i]->output;
				outputLayer->neurons[i]->delta = (expected[i] - out) * sigmoidDerivative(out);
			}

			// Compute delta for hidden layers
			for (int l = num_layers - 2; l >= 0; --l) {
				Layer* current = layers[l];
				Layer* next = layers[l + 1];
				for (int i = 0; i < current->num_neurons; ++i) {
					double error = 0.0;
					for (int j = 0; j < next->num_neurons; ++j) {
						error += next->neurons[j]->weights[i] * next->neurons[j]->delta;
					}
					current->neurons[i]->delta = error * sigmoidDerivative(current->neurons[i]->output);
				}
			}

			// Update weights and biases
			for (int l = 0; l < num_layers; ++l) {
				Layer* layer = layers[l];
				double* layerInputs;
				int num_inputs;

				if (l == 0) {
					layerInputs = inputs;
					num_inputs = input_size;
				}
				else {
					Layer* prevLayer = layers[l - 1];
					num_inputs = prevLayer->num_neurons;
					layerInputs = new double[num_inputs];
					for (int i = 0; i < num_inputs; ++i)
						layerInputs[i] = prevLayer->neurons[i]->output;
				}

				for (int n = 0; n < layer->num_neurons; ++n) {
					Neuron* neuron = layer->neurons[n];
					for (int w = 0; w < neuron->num_weights; ++w) {
						neuron->weights[w] += learningRate * neuron->delta * layerInputs[w];
					}
					neuron->bias += learningRate * neuron->delta;
				}

				if (l != 0) delete[] layerInputs;
			}

			delete[] outputs;
		}

		void clearData() {
			// Optional: Reset outputs/deltas if needed
		}
};

int validateInput(int low, int high);

int main() {
    cout << "*****************************************************\n";
    cout << "==== STARTING NEURAL NETWORK APPLICATION PROGRAM ====\n";
    cout << "*****************************************************\n";

    bool running = true;
    int input;

    cout << "WHAT NEURAL NETWORK APPLICATION WOULD YOU LIKE TO USE:\n";
    cout << "Type '1' for XOR Logic Gate Neural Network\n";
    cout << "Type '2' for Binary -> Decimal Converter\n";
    cout << "Type '3' for Odd or Even Number Classifier\n";
    cout << "Type '4' for Prime Number Detector\n";
    cout << "Type '5' for Linear Equation Solver\n";
    cout << "Type '6' to Exit Program\n";

    int *topology = nullptr;
    NeuralNetwork *nn = nullptr;
	
	double input_data_one;
	double input_data_two;
	double input_data_three; // BINARY-DECIMAL CONVERTER
	double input_data_four; // BINARY-DECIMAL-CONVERTER
	double* output;
	
	// XOR SPECIFIC
	double training_inputs_XOR[4][2] = 
	{
		{0,0}, {0,1}, {1,0}, {1,1}
	};
	double expected_outputs_XOR[4][1] = 
	{
		{0}, {1}, {1}, {0}
	};
	double input_set_XOR[1][2];
	
	
	// BINARY-DECIMAL SPECIFIC
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
	
	
	// ------------------ MAIN APPLICATION LOOP ------------------ //
    do {
        cout << "Input: ";
        cin >> input;
		cout << '\n';
		
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        switch (input) {
        case 1: // XOR Logic Gate Neural Network
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
			input_data_one = validateInput(0,1);
			cout << "SECOND VALUE: \n";
			input_data_two = validateInput(0,1);
			
			input_set_XOR[0][0] = input_data_one;
			input_set_XOR[0][1] = input_data_two;
			
			// Predict Results
			output = nn->predict(input_set_XOR[0],2);
			cout << "XOR Result: " << output[0] << "\n";
            break;
        case 2: // BINARY-DECIMAL-CONVERTER
			// Setup Variables: Layers
			topology = new int[3];
			topology[0] = 4; topology[1] = 8; topology[2] = 1;
			nn = new NeuralNetwork(topology,3);
			
			// Adjusting weights to be accurate
			for(int epoch = 0; epoch < 30000; ++epoch){
				for(int i = 0; i < 16; ++i)
				{
					nn->train(training_inputs_BD[i], expected_outputs_BD[i], 4);
				}
			}
			
			// Asking for User Input
			cout << "FIRST VALUE: \n";
			input_data_one = validateInput(0,1);
			cout << "SECOND VALUE: \n";
			input_data_two = validateInput(0,1);
			cout << "THIRD VALUE: \n";
			input_data_three = validateInput(0,1);
			cout << "FOURTH VALUE: \n";
			input_data_four = validateInput(0,1);
			
			// make sure to round up btw c:
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
        case 3:
            break;
        case 4:
            break;
        case 5:
            break;
        case 6:
            running = false;
            break;
        default:
            cout << "Error! Please input a number from 1-6.\n";
            break;
        };
		
		// Cleanup Memory
		delete nn;
		delete[] topology;
		delete[] output;
		output = nullptr;
		topology = nullptr;
		nn = nullptr;
		
		cout << '\n';
    } while (running);

    cout << "*****************************************************\n";
    cout << "===== ENDING NEURAL NETWORK APPLICATION PROGRAM =====\n";
    cout << "*****************************************************\n";

    getch();
    return 0;
}

int validateInput(int low, int high){
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