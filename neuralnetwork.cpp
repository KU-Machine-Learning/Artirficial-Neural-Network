#include "neuralnetwork.h"

NeuralNetwork::NeuralNetwork(Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> adjacencyMatrix, std::function<float (float)> activationFunction, std::function<float (float)> activationFunctionPrime, int numInputs, int numOutputs)
{
	if(adjacencyMatrix.rows() != adjacencyMatrix.cols())
	{
		throw std::invalid_argument("adjacency matrix must be square");
	}

	m_activationFunction = activationFunction;
	m_activationFunctionPrime = activationFunctionPrime;

	m_size = adjacencyMatrix.rows();
	m_numInputs = numInputs;
	m_numOutputs = numOutputs;

	m_weights.resize(adjacencyMatrix.rows(), adjacencyMatrix.rows());
	for(int i = 0; i < adjacencyMatrix.rows(); i++)
	{
		for(int j = 0; j < adjacencyMatrix.cols(); j++)
		{
			if(adjacencyMatrix(i, j))
			{
				m_weights.insert(i, j) = randomWeight(-1.0, 1.0);
			}
		}
	}

	m_thresholds.resize(adjacencyMatrix.rows());
	for(int i = 0; i < adjacencyMatrix.rows(); i++)
	{
		m_thresholds(i) = randomWeight(-1.0, 1.0);
	}
}

std::vector<float> NeuralNetwork::evaluate(std::vector<float> inputs)
{
	Eigen::SparseVector<float> outputs(m_size);
	// initialize sparse vector with inputs
	for(int i = 0; i < inputs.size(); i++)
	{
		outputs.insert(i) = inputs[i];
	}

	for(int i = inputs.size(); i < m_size; i++)
	{
		float net = outputs.dot(m_weights.col(i)) + m_thresholds(i);
		outputs.insert(i) = m_activationFunction(net);
	}

	std::vector<float>* networkOutputs = new std::vector<float>(m_numOutputs);
	for(int i = 0; i < m_numOutputs; i++)
	{
		(*networkOutputs)[i] = outputs.coeff(m_size - m_numOutputs + i);
	}

	return *networkOutputs;
}

void NeuralNetwork::learn(std::vector<float> inputs, std::vector<float> expectedOutputs)
{
	// outputs is the standard output of each neuron; outputsPrime is similar
	// to outputs, but is calculated with m_activationFuctionPrime instead.
	// outputsPrime is used in delta calculations during backpropogation, but
	// it is calculated here
	Eigen::SparseVector<float> outputs(m_size);
	Eigen::SparseVector<float> outputsPrime(m_size);

	// initialize sparse vector with inputs
	for(int i = 0; i < inputs.size(); i++)
	{
		outputs.insert(i) = inputs[i];
	}

	// feedforward
	for(int i = inputs.size(); i < m_size; i++)
	{
		float net = outputs.dot(m_weights.col(i)) + m_thresholds(i);
		outputs.insert(i) = m_activationFunction(net);
		outputsPrime.insert(i) = m_activationFunctionPrime(net);
	}

	Eigen::SparseVector<float> deltas(m_size);

	// backpropogation starts by assigning output neuron deltas, comparing
	// outputs to expectedOutputs
	for(int i = 0; i < m_numOutputs; i++)
	{
		float difference = expectedOutputs[i] - outputs.coeff(m_size - m_numOutputs + i);
		deltas.insert(m_size - m_numOutputs + i) = difference * outputsPrime.coeff(m_size - m_numOutputs + i);
	}

	// actual backpropagation algorithm
	for(int i = m_size - m_numOutputs - 1; i > m_numInputs - 1; i--)
	{
		float weightedDeltas = deltas.dot(m_weights.row(i));
		deltas.insert(i) = weightedDeltas * outputsPrime.coeff(i);
	}

	// weight update
	// note: this happens in no particular order, since after deltas have been
	// calculated, weight updates can occur in any order.  This iterator
	// efficiently iterates through the sparse matrix of weights.
	for (int k = 0; k < m_weights.outerSize(); ++k)
	{
		for (Eigen::SparseMatrix<float>::InnerIterator it(m_weights, k); it; ++it)
		{
			float change = deltas.coeff(it.col()) * outputs.coeff(it.row()) * .1;
			m_weights.coeffRef(it.row(), it.col()) = it.value() + change;
		}
	}

	// threshold weight updates
	for(int i = m_numInputs; i < m_size; i++)
	{
		float change = deltas.coeff(i) * .1;
		m_thresholds(i) = m_thresholds(i) + change;
	}

}

// This code is straight-up copied from stack overflow
float NeuralNetwork::randomWeight(float min, float max)
{
	return min + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(max-min)));
}

Eigen::SparseMatrix<float> NeuralNetwork::getWeights()
{
	return m_weights;
}
