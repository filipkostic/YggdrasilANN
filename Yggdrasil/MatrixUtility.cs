using MathNet.Numerics.LinearAlgebra;
using System;
using System.Linq;

namespace NeuralNetwork
{
    static class MatrixUtility
    {
        internal static Tuple<Matrix<T>, Matrix<T>> Split<T>(this Matrix<T> matrix, int[] indexes)
            where T : struct, IEquatable<T>, IFormattable
        {
            Matrix<T> first = Matrix<T>.Build.Dense(matrix.RowCount - indexes.Length, matrix.ColumnCount);
            Matrix<T> second = Matrix<T>.Build.Dense(indexes.Length, matrix.ColumnCount);
            int firstIndex = 0;
            int secondIndex = 0;
            for (int i = 0; i < matrix.RowCount; ++i)
            {
                if (indexes.Contains(i))
                {
                    second.SetRow(secondIndex++, matrix.Row(i));
                }
                else
                {
                    first.SetRow(firstIndex++, matrix.Row(i));
                }
            }
            return new Tuple<Matrix<T>, Matrix<T>>(first, second);
        }

        internal static int GetSizeFromPercentage<T>(this Matrix<T> matrix, double percentage)
            where T : struct, IEquatable<T>, IFormattable
        {
            return (int)Math.Floor(matrix.RowCount / percentage);
        }

        internal static Matrix<double> Random(this MatrixBuilder<double> builder, int rows, int columns)
        {
            Random randomizer = new Random();
            Matrix<double> weights = Matrix<double>.Build.Dense(rows, columns);
            for (int row = 0; row < rows; ++row)
            {
                for (int column = 0; column < columns; ++column)
                {
                    weights[row, column] = randomizer.NextDouble() * 0.24 - 0.12;
                }
            }
            return weights;
        }

        public static double SumOfSquaredMatrix(Matrix<double> firstLayerWeights, Matrix<double> secondLayerWeights)
        {
            Matrix<double> temporaryWeightsFirst = Matrix<double>.Build.DenseOfMatrix(firstLayerWeights);
            temporaryWeightsFirst.SetColumn(0, Vector<double>.Build.Dense(temporaryWeightsFirst.RowCount, 0d));
            Matrix<double> temporaryWeightsSecond = Matrix<double>.Build.DenseOfMatrix(secondLayerWeights);
            temporaryWeightsSecond.SetColumn(0, Vector<double>.Build.Dense(temporaryWeightsSecond.RowCount, 0d));
            return temporaryWeightsFirst.PointwisePower(2d).ColumnSums().Sum() + temporaryWeightsSecond.PointwisePower(2d).ColumnSums().Sum();
        }

        public static Vector<double> UnrollMatrices(Matrix<double> inputLayerWeights, Matrix<double> hiddenLayerWeights)
        {
            Vector<double> unrolledGradient = Vector<double>.Build.Dense(inputLayerWeights.ColumnCount * inputLayerWeights.RowCount + hiddenLayerWeights.RowCount * hiddenLayerWeights.ColumnCount);
            unrolledGradient.SetSubVector(0, inputLayerWeights.ColumnCount * inputLayerWeights.RowCount, Vector<double>.Build.DenseOfArray(inputLayerWeights.ToColumnWiseArray()));
            unrolledGradient.SetSubVector(inputLayerWeights.ColumnCount * inputLayerWeights.RowCount, hiddenLayerWeights.RowCount * hiddenLayerWeights.ColumnCount, Vector<double>.Build.DenseOfArray(hiddenLayerWeights.ToColumnWiseArray()));
            return unrolledGradient;
        }

        public static Tuple<Matrix<double>, Matrix<double>> ReshapeMatrices<T>(Vector<double> unrolledMatrices, int inputLayerWeightsRows, int inputLayerWeightsColumns, int hiddenLayerWeightsRows, int hiddenLayerWeightsColumns)
        {
            return new Tuple<Matrix<double>, Matrix<double>>(
                Matrix<double>.Build.DenseOfColumnMajor(inputLayerWeightsRows, inputLayerWeightsColumns, unrolledMatrices.SubVector(0, inputLayerWeightsRows * inputLayerWeightsColumns)),
                Matrix<double>.Build.DenseOfColumnMajor(hiddenLayerWeightsRows, hiddenLayerWeightsColumns, unrolledMatrices.SubVector(inputLayerWeightsRows * inputLayerWeightsColumns, hiddenLayerWeightsRows * hiddenLayerWeightsColumns)));
        }
    }
}
