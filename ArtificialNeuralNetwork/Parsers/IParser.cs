using MathNet.Numerics.LinearAlgebra;
using System;

namespace ArtificialNeuralNetwork.Parsers
{
    interface IParser
    {
        Tuple<Matrix<double>, Matrix<double>> Read(string path);
    }
}
