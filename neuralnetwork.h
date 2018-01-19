#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <stdexcept>
#include <vector>
#include <functional>
#include "Eigen/Dense"
#include "Eigen/Sparse"

class NeuralNetwork
{
public:
	NeuralNetwork(Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>, std::function<float (float)>, std::function<float (float)>, int, int);
	std::vector<float> evaluate(std::vector<float>);
	void learn(std::vector<float>, std::vector<float>);
	float randomWeight(float, float);
	Eigen::SparseMatrix<float> getWeights();

private:
	std::function<float (float)> m_activationFunction;
	std::function<float (float)> m_activationFunctionPrime;
	Eigen::SparseMatrix<float> m_weights;
	Eigen::VectorXf m_thresholds;
	int m_size;
	int m_numInputs;
	int m_numOutputs;
};

#endif
