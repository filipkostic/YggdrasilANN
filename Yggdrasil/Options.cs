using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetwork
{
    class Options
    {
        public Options(DataSet training, DataSet test, Tuple<Matrix<double>, Matrix<double>> weights, int epochs, double lambda)
        {
            Training = training;
            Test = test;
            Weights = weights;
            Epochs = epochs;
            Lambda = lambda;
        }
        public double Lambda { get; }
        public int Epochs { get; }
        public DataSet Training { get; }
        public Tuple<Matrix<double>, Matrix<double>> Weights { get; set; }
        public DataSet Test { get; }
    }
}
