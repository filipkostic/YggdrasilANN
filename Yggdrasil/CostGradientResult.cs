using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class CostGradientResult : ICostGradientResult
    {
        public CostGradientResult(double cost, Vector<double> gradient, double accuracy)
        {
            Cost = cost;
            Gradient = gradient;
            Accuracy = accuracy;
        }

        public double Cost { get; }

        public Vector<double> Gradient { get; }

        public double Accuracy { get; }
    }
}
