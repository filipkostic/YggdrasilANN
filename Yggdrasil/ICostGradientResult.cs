using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    public interface ICostGradientResult
    {
        double Cost { get; }
        Vector<double> Gradient { get; }
    }
}
