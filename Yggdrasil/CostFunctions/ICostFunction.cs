using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.CostFunctions
{
    public interface ICostFunction
    {
        ICostGradientResult CostAndGradient(Matrix<double> currentLayerWeights, Matrix<double> nextLayerWeights, Matrix<double> inputs, Matrix<double> desiredOutputs, double lambda);
        void FeedForward(Matrix<double> inputWeights, Matrix<double> hiddenLayerWeights, Matrix<double> inputs, int i, out Vector<double> a1, out Vector<double> z2, out Vector<double> a2, out Vector<double> a3);
        double[] FeedForward(double[] unrolledWeights, int numberOfHiddenNeurons, int numberOfOutputs, double[] inputs);
        double CalculateAccuracy(Matrix<double> testSet, Matrix<double> testSetIdealOutput, Matrix<double> inputLayerWeights, Matrix<double> hiddenLayerWeights);
    }
}
