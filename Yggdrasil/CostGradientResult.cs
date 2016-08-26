using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class CostGradientResult : ICostGradientResult
    {
        public CostGradientResult(double cost, Vector<double> gradient)
        {
            Cost = cost;
            lock (costHistory)
            {
                costHistory.Add(cost);
            }
            Gradient = gradient;
        }

        static List<double> costHistory = new List<double>();
        public static List<double> History
        {
            get
            {
                return costHistory.Select(x => x).ToList();
            }
        }
        public List<double> CostHistory
        {
            get
            {
                return costHistory.Select(x => x).ToList();
            }
        }

        public double Cost { get; }

        public Vector<double> Gradient { get; }
    }
}
