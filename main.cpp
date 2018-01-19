#include <iostream>
#include <vector>
#include <math.h>
#include <sys/resource.h>
#include "Eigen/Dense"

#include "neuralnetwork.h"

using namespace Eigen;

Matrix<bool, Dynamic, Dynamic> fullyConnectedShallow(const int numInputs, const int numHidden, const int numOutputs)
{
	const int size = numInputs + numHidden + numOutputs;

	Matrix<bool, Dynamic, Dynamic> adjacencyMatrix(size, size);

	for(int i = 0; i < numInputs; i++)
	{
		for(int j = numInputs; j < numInputs + numHidden; j++)
		{
			adjacencyMatrix(i, j) = true;
		}
	}

	for(int i = numInputs; i < numInputs + numHidden; i++)
	{
		for(int j = numInputs + numHidden; j < size; j++)
		{
			adjacencyMatrix(i, j) = true;
		}
	}

	return adjacencyMatrix;
}

Matrix<bool, Dynamic, Dynamic> perceptron(const int numInputs)
{
	const int size = numInputs + 1;

	Matrix<bool, Dynamic, Dynamic> adjacencyMatrix(size, size);

	for(int i = 0; i < numInputs; i++)
	{
		adjacencyMatrix(i, size - 1) = true;
	}

	return adjacencyMatrix;
}

float ReLU(float x)
{
	if(x > 0)
	{
		return x;
	}
	else
	{
		return 0;
	}
}

float ReLU_prime(float x)
{
	if(x > 0)
	{
		return 1.0;
	}
	else
	{
		return 0.1;
	}
}

float mytanh(float x)
{
	return (expf(x) - expf(-x))/(expf(x) + expf(-x));
}

float tanh_prime(float x)
{
	return 2.0 / (expf(x) + expf(-x));
}

int main()
{
	// this is also stack overflow code
	srand (static_cast <unsigned> (time(0)));

	Matrix<bool, Dynamic, Dynamic> mat1 = fullyConnectedShallow(2, 200, 2);
	//Matrix<bool, Dynamic, Dynamic> mat1 = perceptron(2);

	NeuralNetwork net(mat1, ReLU, ReLU_prime, 2, 2);

	std::cout << net.getWeights() << std::endl;

	std::vector<float> inputs1 = {1.0, 1.0};
	std::vector<float> outputs1 = {1.0, 0.0};
	std::vector<float> inputs2 = {-1.0, -1.0};
	std::vector<float> outputs2 = {1.0, 0.0};
	std::vector<float> inputs3 = {1.0, -1.0};
	std::vector<float> outputs3 = {0.0, 1.0};
	std::vector<float> inputs4 = {-1.0, 1.0};
	std::vector<float> outputs4 = {0.0, 1.0};
	for(int i = 0; i < 200; i++)
	{
		std::cout << "Iteration " << i << std::endl;
		net.learn(inputs1, outputs1);
		net.learn(inputs2, outputs2);
		net.learn(inputs3, outputs3);
		net.learn(inputs4, outputs4);
	}

	std::cout << net.getWeights() << std::endl;

	std::vector<float> outputs = net.evaluate(inputs1);
	std::cout << "Network Outputs (ex1):" << std::endl;
	for(int i = 0; i < outputs.size(); i++)
	{
		std::cout << outputs[i] << ' ';
	}
	std::cout << std::endl;

	outputs = net.evaluate(inputs2);
	std::cout << "Network Outputs (ex2):" << std::endl;
	for(int i = 0; i < outputs.size(); i++)
	{
		std::cout << outputs[i] << ' ';
	}
	std::cout << std::endl;

	outputs = net.evaluate(inputs3);
	std::cout << "Network Outputs (ex3):" << std::endl;
	for(int i = 0; i < outputs.size(); i++)
	{
		std::cout << outputs[i] << ' ';
	}
	std::cout << std::endl;

	outputs = net.evaluate(inputs4);
	std::cout << "Network Outputs (ex4):" << std::endl;
	for(int i = 0; i < outputs.size(); i++)
	{
		std::cout << outputs[i] << ' ';
	}
	std::cout << std::endl;

	return 0;
}
