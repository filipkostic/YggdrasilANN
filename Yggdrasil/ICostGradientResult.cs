using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    interface ICostGradientResult
    {
        double Cost { get; }
        Vector<double> Gradient { get; }
    }
}
