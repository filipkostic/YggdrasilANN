using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    interface ICostGradientTuple
    {
        double Cost { get; }
        Vector<double> Gradient { get; }
    }
}
