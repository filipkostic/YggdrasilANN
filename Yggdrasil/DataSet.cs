using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    class DataSet
    {
        public DataSet(Matrix<double> set, Matrix<double> desired)
        {
            Set = set;
            Desired = desired;
        }
        public Matrix<double> Set { get; }
        public Matrix<double> Desired { get; }
    }
}
