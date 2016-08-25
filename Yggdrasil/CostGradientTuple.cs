using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    class CostGradientTuple : ICostGradientTuple
    {
        public CostGradientTuple(double cost, Vector<double> gradient)
        {
            Cost = cost;
            Gradient = gradient;
        }

        public double Cost { get; }

        public Vector<double> Gradient { get; }
    }
}
