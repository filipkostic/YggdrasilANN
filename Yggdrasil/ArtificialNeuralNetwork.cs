using MathNet.Numerics.LinearAlgebra;
using System;
using Logger;
using System.Linq;
using NeuralNetwork.CostFunctions;

namespace NeuralNetwork
{
    public class ArtificialNeuralNetwork
    {
        Matrix<double> TrainingSet { get; }
        Matrix<double> TrainingSetDesiredOutput { get; }
        Matrix<double> TestSet { get; }
        Matrix<double> TestSetDesiredOutput { get; }
        int NeuronsInHiddenLayer { get; }
        int Epochs { get; }
        double Lambda { get; }
        ICostFunction CostFunction { get; }
        Lazy<ANNLogger> logger = new Lazy<ANNLogger>(() => new ANNLogger(
            String.Format("{0} - {1}.json", DateTime.Now.ToString("yyyy-MM-dd HH-mm-ss"), System.Threading.Thread.CurrentThread.ManagedThreadId)));
        ANNLogger Logger
        {
            get
            {
                return logger.Value;
            }
        }

        ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, int hiddenLayer, int epochs, double lambda, CostFunctionTypes costFunction)
        {
            TrainingSet = set;
            TrainingSetDesiredOutput = desiredOutput;
            TestSet = testSet;
            TestSetDesiredOutput = testSetDesiredOutput;
            NeuronsInHiddenLayer = hiddenLayer;
            Epochs = epochs;
            Lambda = lambda;
            CostFunction = CostFunctions.CostFunction.Build(costFunction);
        }

        ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, int hiddenLayer, int epochs, CostFunctionTypes costFunction)
            : this(set, desiredOutput, testSet, testSetDesiredOutput, hiddenLayer, epochs, 1d, costFunction)
        {

        }

        ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, CostFunctionTypes costFunction)
            : this(set, desiredOutput, testSet, testSetDesiredOutput, 25, 50, costFunction)
        {

        }

        ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, int hiddenLayer, int epochs, double lambda, CostFunctionTypes costFunction)
        {
            var testSetIndexes = Utility.UniqueRandomArray(set.RowCount - 1, set.GetSizeFromPercentage(0.3d));
            var sets = set.Split(testSetIndexes);
            TrainingSet = sets.Item1;
            TestSet = sets.Item2;
            var desiredOutputs = desiredOutput.Split(testSetIndexes);
            TrainingSetDesiredOutput = desiredOutputs.Item1;
            TestSetDesiredOutput = desiredOutputs.Item2;
            NeuronsInHiddenLayer = hiddenLayer;
            Epochs = epochs;
            Lambda = lambda;
            CostFunction = CostFunctions.CostFunction.Build(costFunction);
        }

        ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput, int hiddenLayer, int epochs, CostFunctionTypes costFunction)
            : this(set, desiredOutput, hiddenLayer, epochs, 1d, costFunction)
        {

        }

        ArtificialNeuralNetwork(Matrix<double> set, Matrix<double> desiredOutput)
            : this(set, desiredOutput, 25, 50, CostFunctionTypes.Sigmoid)
        {

        }

        public static ArtificialNeuralNetwork Build(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, int hiddenLayer, int epochs, double lambda, CostFunctionTypes costFunction)
        {
            return new ArtificialNeuralNetwork(set, desiredOutput, testSet, testSetDesiredOutput, hiddenLayer, epochs, lambda, costFunction);
        }

        public static ArtificialNeuralNetwork Build(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, int hiddenLayer, int epochs, CostFunctionTypes costFunction)
        {
            return new ArtificialNeuralNetwork(set, desiredOutput, testSet, testSetDesiredOutput, hiddenLayer, epochs, costFunction);
        }

        public static ArtificialNeuralNetwork Build(Matrix<double> set, Matrix<double> desiredOutput, Matrix<double> testSet, Matrix<double> testSetDesiredOutput, CostFunctionTypes costFunction)
        {
            return new ArtificialNeuralNetwork(set, desiredOutput, testSet, testSetDesiredOutput, costFunction);
        }

        public static ArtificialNeuralNetwork Build(Matrix<double> set, Matrix<double> desiredOutput, int hiddenLayer, int epochs, double lambda, CostFunctionTypes costFunction)
        {
            return new ArtificialNeuralNetwork(set, desiredOutput, hiddenLayer, epochs, lambda, costFunction);
        }

        public static ArtificialNeuralNetwork Build(Matrix<double> set, Matrix<double> desiredOutput, int hiddenLayer, int epochs, CostFunctionTypes costFunction)
        {
            return new ArtificialNeuralNetwork(set, desiredOutput, hiddenLayer, epochs, costFunction);
        }

        public static ArtificialNeuralNetwork Build(Matrix<double> set, Matrix<double> desiredOutput)
        {
            return new ArtificialNeuralNetwork(set, desiredOutput);
        }

        public ICostGradientResult Learn()
        {
            var ann = Fmincg.Build(CostFunction);
            ann.LearningEvent += Ann_LearningEvent;
            var weights = InitializeWeightLayers();
            Logger.Start(TrainingSet.RowCount, TestSet.RowCount, Epochs, Lambda, NeuronsInHiddenLayer, "sigmoid");
            var result = ann.MinimizeFunction(
                MatrixUtility.UnrollMatrices(weights.Item1, weights.Item2),
                new Options(
                    new DataSet(TrainingSet, TrainingSetDesiredOutput),
                    new DataSet(TestSet, TestSetDesiredOutput),
                    weights,
                    Epochs,
                    Lambda));
            Logger.Finish(result.ToArray());
            weights = result.ReshapeMatrices(weights.Item1.RowCount, weights.Item1.ColumnCount, weights.Item2.RowCount, weights.Item2.ColumnCount);
            double accuracy = CostFunction.CalculateAccuracy(TestSet, TestSetDesiredOutput, weights.Item1, weights.Item2);
            return new CostGradientResult(0, result, accuracy);
        }

        void Ann_LearningEvent(int epoch, Options options, Vector<double> input, double cost)
        {
            var weights = input.ReshapeMatrices(options.Weights.Item1.RowCount, options.Weights.Item1.ColumnCount, options.Weights.Item2.RowCount, options.Weights.Item2.ColumnCount);
            double accuracy = CostFunction.CalculateAccuracy(options.Test.Set, options.Test.Desired, weights.Item1, weights.Item2);
            Logger.AddEpoch(epoch, cost, accuracy);
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
