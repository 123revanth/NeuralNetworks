#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
public:
	NeuralNetwork(std::vector<uint> topology, Scalar learningRate = Scalar(0.005));
	void propagateForward(RowVector& input);
	void propagateBackward(RowVector& output);
	void calcErrors(RowVector& output);

	void updateWeights();
	void train(std::vector<RowVector*> data);

	std::vector<RowVector*> neuronLayers; // stores the different layers of out network
	std::vector<RowVector*> cacheLayers; // stores the unactivated (activation fn not yet applied) values of layers
	std::vector<RowVector*> deltas; // stores the error contribution of each neurons
	std::vector<Matrix*> weights; // the connection weights itself
	Scalar learningRate;
};
