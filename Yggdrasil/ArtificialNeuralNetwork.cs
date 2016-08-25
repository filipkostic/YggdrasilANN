using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class ArtificialNeuralNetwork
    {
        Matrix<double> TrainingSet { get; }
        Matrix<double> TrainingSetDesiredOutput { get; }
        Matrix<double> TestSet { get; }
        Matrix<double> TestSetDesiredOutput { get; }
        int NeuronsInHiddenLayer { get; }
        int Epochs { get; }
        double Lambda { get; }

        public ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, int hiddenLayer, int epochs, double lambda)
        {
            TrainingSet = set;
            TrainingSetDesiredOutput = desiredOutput;
            TestSet = testSet;
            TestSetDesiredOutput = testSetDesiredOutput;
            NeuronsInHiddenLayer = hiddenLayer;
            Epochs = epochs;
            Lambda = lambda;
        }

        public ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, int hiddenLayer, int epochs)
            : this(set, desiredOutput, testSet, testSetDesiredOutput, hiddenLayer, epochs, 1d)
        {

        }

        public ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput)
            : this(set, desiredOutput, testSet, testSetDesiredOutput, 25, 50)
        {

        }

        public ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, int hiddenLayer, int epochs, double lambda)
        {
            var testSetIndexes = Utility.UniqueRandomArray(set.RowCount - 1, set.GetSizeFromPercentage(30d));
            var sets = set.Split(testSetIndexes);
            TrainingSet = sets.Item1;
            TestSet = sets.Item2;
            var desiredOutputs = desiredOutput.Split(testSetIndexes);
            TrainingSetDesiredOutput = desiredOutputs.Item1;
            TestSetDesiredOutput = desiredOutputs.Item2;
            NeuronsInHiddenLayer = hiddenLayer;
            Epochs = epochs;
            Lambda = lambda;
        }

        public ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, int hiddenLayer, int epochs)
            : this(set, desiredOutput, hiddenLayer, epochs, 1d)
        {

        }

        public ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput)
            : this(set, desiredOutput, 25, 50)
        {

        }

        public ICostGradientTuple Learn()
        {
            var weights = InitializeWeightLayers();
            var result = Fmincg.MinimizeFunction(
                CostFunctions.Sigmoid.CostAndGradient,
                MatrixUtility.UnrollMatrices(weights.Item1, weights.Item2),
                new Options(
                    new DataSet(TrainingSet, TrainingSetDesiredOutput),
                    new DataSet(TestSet, TestSetDesiredOutput),
                    weights,
                    Epochs,
                    Lambda));
            throw new System.NotImplementedException();
        }

        Tuple<Matrix<double>, Matrix<double>> InitializeWeightLayers()
        {
            return new Tuple<Matrix<double>, Matrix<double>>(
                InitializeWeightLayer(NeuronsInHiddenLayer, TrainingSet.ColumnCount),
                InitializeWeightLayer(TrainingSetDesiredOutput.ColumnCount, NeuronsInHiddenLayer));
        }

        Matrix<double> InitializeWeightLayer(int currentLayer, int previousLayer)
        {
            return Matrix<double>.Build.Random(currentLayer, previousLayer + 1);
        }
    }
}
