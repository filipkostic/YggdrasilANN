using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetwork.CostFunctions
{
    class Sigmoid
    {
        static double SigmoidFunction(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static Matrix<double> ActivationFunction(Matrix<double> z)
        {
            Matrix<double> temporaryLayer = Matrix<double>.Build.DenseOfMatrix(z);
            for (int row = 0; row < z.RowCount; ++row)
            {
                for (int column = 0; column < z.ColumnCount; ++column)
                {
                    temporaryLayer[row, column] = SigmoidFunction(z[row, column]);
                }
            }
            return temporaryLayer;
        }

        public static Vector<double> ActivationFunction(Vector<double> z)
        {
            return z.Map(x => SigmoidFunction(x));
        }

        public static Vector<double> Gradient(Vector<double> z)
        {
            Vector<double> sigmoidLayer = ActivationFunction(z);
            return sigmoidLayer.PointwiseMultiply(1 - sigmoidLayer);
        }

        public static ICostGradientResult CostAndGradient(Matrix<double> currentLayerWeights, Matrix<double> nextLayerWeights, Matrix<double> inputs, Matrix<double> desiredOutputs, double lambda)
        {
            Matrix<double> thetaGrad1 = Matrix<double>.Build.Dense(currentLayerWeights.RowCount, currentLayerWeights.ColumnCount, 0d);
            Matrix<double> thetaGrad2 = Matrix<double>.Build.Dense(nextLayerWeights.RowCount, nextLayerWeights.ColumnCount, 0d);
            Matrix<double> A3;
            double regularization;
            CalculateRegularization(currentLayerWeights, nextLayerWeights, inputs, lambda, out A3, out regularization);
            double cost = CalculateCost(A3, desiredOutputs, regularization);
            for (int i = 0; i < inputs.RowCount; ++i)
            {
                Vector<double> a1, z2, a2, a3;
                FeedForward(currentLayerWeights, nextLayerWeights, inputs, i, out a1, out z2, out a2, out a3);
                BackPropagation(nextLayerWeights, desiredOutputs, ref thetaGrad1, ref thetaGrad2, i, a1, z2, a2, a3);
            }
            Vector<double> unrolledGradient = CalculateGradientAndUnroll(currentLayerWeights, nextLayerWeights, desiredOutputs, lambda, ref thetaGrad1, ref thetaGrad2);
            return new CostGradientResult(cost, unrolledGradient);
        }

        static void CalculateRegularization(Matrix<double> inputWeights, Matrix<double> hiddenLayerWeights, Matrix<double> inputs, double lambda, out Matrix<double> A3, out double regularization)
        {
            Matrix<double> temporaryInput = Matrix<double>.Build.DenseOfMatrix(inputs);
            Matrix<double> A1 = temporaryInput.InsertColumn(0, Vector<double>.Build.Dense(inputs.RowCount, 1d));
            Matrix<double> A2 = ActivationFunction(A1 * inputWeights.Transpose());
            A3 = ActivationFunction(A2.InsertColumn(0, Vector<double>.Build.Dense(A2.RowCount, 1d)) * hiddenLayerWeights.Transpose());
            regularization = lambda * MatrixUtility.SumOfSquaredMatrix(inputWeights, hiddenLayerWeights) / (2 * inputs.RowCount);
        }

        static Vector<double> CalculateGradientAndUnroll(Matrix<double> inputWeights, Matrix<double> hiddenLayerWeights, Matrix<double> desiredOutputs, double lambda, ref Matrix<double> thetaGrad1, ref Matrix<double> thetaGrad2)
        {
            Matrix<double> temporary = Matrix<double>.Build.DenseOfMatrix(inputWeights);
            temporary.SetColumn(0, Vector<double>.Build.Dense(temporary.RowCount, 0d));
            thetaGrad1 = thetaGrad1 / desiredOutputs.RowCount + lambda / desiredOutputs.RowCount * temporary;
            temporary = Matrix<double>.Build.DenseOfMatrix(hiddenLayerWeights);
            temporary.SetColumn(0, Vector<double>.Build.Dense(temporary.RowCount, 0d));
            thetaGrad2 = thetaGrad2 / desiredOutputs.RowCount + lambda / desiredOutputs.RowCount * temporary;
            Vector<double> unrolledGradient = MatrixUtility.UnrollMatrices(thetaGrad1, thetaGrad2);
            return unrolledGradient;
        }

        static double CalculateCost(Matrix<double> outputs, Matrix<double> desiredOutputs, double regularization = 0)
        {
            double cost = 0d;
            Matrix<double> temporary = desiredOutputs.PointwiseMultiply(outputs.PointwiseLog())
                + (1 - desiredOutputs).PointwiseMultiply((1 - outputs).PointwiseLog());
            cost = -temporary.RowSums().Sum() / desiredOutputs.RowCount + regularization;
            return cost;
        }

        static void BackPropagation(Matrix<double> hiddenLayerWeights, Matrix<double> desiredOutputs, ref Matrix<double> thetaGrad1, ref Matrix<double> thetaGrad2, int i, Vector<double> a1, Vector<double> z2, Vector<double> a2, Vector<double> a3)
        {
            Vector<double> delta3 = a3 - desiredOutputs.Row(i);
            Vector<double> temporaryVector = Vector<double>.Build.DenseOfVector(hiddenLayerWeights.Transpose() * delta3);
            Vector<double> delta2 = temporaryVector.SubVector(1, temporaryVector.Count - 1).PointwiseMultiply(Gradient(z2));
            thetaGrad1 = thetaGrad1.Add(delta2.OuterProduct(a1));
            thetaGrad2 = thetaGrad2.Add(delta3.OuterProduct(a2));
        }

        public static void FeedForward(Matrix<double> inputWeights, Matrix<double> hiddenLayerWeights, Matrix<double> inputs, int i, out Vector<double> a1, out Vector<double> z2, out Vector<double> a2, out Vector<double> a3)
        {
            a1 = Vector<double>.Build.Dense(inputs.ColumnCount + 1, 1d);
            a1.SetSubVector(1, inputs.ColumnCount, inputs.Row(i));
            z2 = inputWeights * a1;
            a2 = Vector<double>.Build.Dense(z2.Count + 1, 1d);
            a2.SetSubVector(1, z2.Count, ActivationFunction(z2));
            a3 = ActivationFunction(hiddenLayerWeights * a2);
        }

        public static double CalculateAccuracy(Matrix<double> testSet, Matrix<double> testSetIdealOutput, Matrix<double> inputLayerWeights, Matrix<double> hiddenLayerWeights)
        {
            double accuracy = 0d;
            for (int i = 0; i < testSet.RowCount; ++i)
            {
                Vector<double> a1, z2, a2, a3;
                FeedForward(inputLayerWeights, hiddenLayerWeights, testSet, i, out a1, out z2, out a2, out a3);
                int ind = testSetIdealOutput.Row(i).MaximumIndex(), ind2 = a3.MaximumIndex();
                if (testSetIdealOutput.Row(i).MaximumIndex() == a3.MaximumIndex())
                {
                    ++accuracy;
                }
            }
            return accuracy / testSet.RowCount;
        }
    }
}
