using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace ContentParsers
{
    public interface IParser
    {
        Tuple<Matrix<double>, Matrix<double>> Read(string path);
    }
}
