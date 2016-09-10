using MathNet.Numerics.LinearAlgebra;
using System;

namespace ContentParsers
{
    public interface IParser
    {
        Tuple<Matrix<double>, Matrix<double>> Read(string path);
    }
}
