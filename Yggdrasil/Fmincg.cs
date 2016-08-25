using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork
{
    internal static class Fmincg
    {
        internal delegate ICostGradientTuple CostFunction(Matrix<double> inputWeights, Matrix<double> hiddenLayerWeights, Matrix<double> inputs, Matrix<double> desiredOutputs, double lambda);
        
        const double EXT = 3.0;
        
        const double RHO = 0.01;
        const double SIG = 0.5;
        const double INT = 0.1;
        const int MAX = 20;
        const int RATIO = 100;

        public static Vector<double> MinimizeFunction(CostFunction costFunction, Vector<double> theta, Options options)
        {
            //if (Log.Logger != null) //TODO
            //{
            //    Log.Logger.Start(options.TrainingSet.RowCount, options.TestSet.RowCount, maxIterations, options.Lambda, options.HiddenLayerWeights.ColumnCount - 1);
            //}
            return Minimize(costFunction, theta, options);
        }

        static Vector<double> Minimize(CostFunction costFunction, Vector<double> theta, Options options)
        {
            Vector<double> input = theta;
            int currentEpoch = 0;
            int red = 1;
            int lineSearchFailures = 0;
            ICostGradientTuple evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
            double cost = evaluate.Cost;
            Vector<double> gradient = evaluate.Gradient;
            //if (Log.Logger != null)
            //{
            //    Matrix<double> inputLayerWeightsReshaped;
            //    Matrix<double> hiddenLayerWeightsReshaped;
            //    Helpers.ReshapeMatrices(input, out inputLayerWeightsReshaped, out hiddenLayerWeightsReshaped, options.InputWeights.RowCount, options.InputWeights.ColumnCount, options.HiddenLayerWeights.RowCount, options.HiddenLayerWeights.ColumnCount);
            //    double accuracy = HelperFunctions.Sigmoid.CalculateAccuracy(options.TestSet, options.TestSetDesiredOutputs, inputLayerWeightsReshaped, hiddenLayerWeightsReshaped);
            //    Log.Logger.AddEpoch(f1, accuracy);
            //}
            currentEpoch = currentEpoch + (options.Epochs < 0 ? 1 : 0);
            Vector<double> s = gradient.Multiply(-1d);

            double slope = s.Multiply(-1d).DotProduct(s);
            double step = red / (1.0 - slope);

            while (EpochsNotFinished(options, currentEpoch))
            {
                currentEpoch = currentEpoch + (options.Epochs > 0 ? 1 : 0);
                Vector<double> inputSet = Vector<double>.Build.DenseOfVector(input);
                double progressCost = cost;
                Vector<double> progressGradient = Vector<double>.Build.DenseOfVector(gradient);
                input = input.Add(s.Multiply(step));
                options.Weights = ReshapeWeights(input, options);
                evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
                double innerCost = evaluate.Cost;
                Vector<double> innerGradient = evaluate.Gradient;
                //if (Log.Logger != null)
                //{
                //    Matrix<double> inputLayerWeightsReshaped;
                //    Matrix<double> hiddenLayerWeightsReshaped;
                //    Helpers.ReshapeMatrices(input, out inputLayerWeightsReshaped, out hiddenLayerWeightsReshaped, options.InputWeights.RowCount, options.InputWeights.ColumnCount, options.HiddenLayerWeights.RowCount, options.HiddenLayerWeights.ColumnCount);
                //    double accuracy = HelperFunctions.Sigmoid.CalculateAccuracy(options.TestSet, options.TestSetDesiredOutputs, inputLayerWeightsReshaped, hiddenLayerWeightsReshaped);
                //    Log.Logger.AddEpoch(f2, accuracy);
                //}

                currentEpoch = currentEpoch + (options.Epochs < 0 ? 1 : 0);
                double d2 = innerGradient.DotProduct(s);
                double f3 = cost;
                double d3 = slope;
                double z3 = -step;
                int M = GetM(options.Epochs, currentEpoch);
                int success = 0;
                double limit = -1;

                while (true)
                {
                    while (((innerCost > cost + step * RHO * slope) | (d2 > -SIG * slope)) && (M > 0))
                    {
                        limit = step;
                        double z2 = 0.0d;
                        double A = 0.0d;
                        double B = 0.0d;
                        if (innerCost > cost)
                        {
                            z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + innerCost - f3);
                        }
                        else
                        {
                            A = 6 * (innerCost - f3) / z3 + 3 * (d2 + d3);
                            B = 3 * (f3 - innerCost) - z3 * (d3 + 2 * d2);
                            z2 = (System.Math.Sqrt(B * B - A * d2 * z3 * z3) - B) / A;
                        }
                        if (IsNotANumber(z2))
                        {
                            z2 = z3 / 2.0d;
                        }
                        z2 = System.Math.Max(System.Math.Min(z2, INT * z3), (1 - INT) * z3);
                        step = step + z2;
                        input = input.Add(s.Multiply(z2));
                        options.Weights = ReshapeWeights(input, options);
                        evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
                        innerCost = evaluate.Cost;
                        innerGradient = evaluate.Gradient;
                        //if (Log.Logger != null)
                        //{
                        //    Matrix<double> inputLayerWeightsReshaped;
                        //    Matrix<double> hiddenLayerWeightsReshaped;
                        //    Helpers.ReshapeMatrices(input, out inputLayerWeightsReshaped, out hiddenLayerWeightsReshaped, options.InputWeights.RowCount, options.InputWeights.ColumnCount, options.HiddenLayerWeights.RowCount, options.HiddenLayerWeights.ColumnCount);
                        //    double accuracy = HelperFunctions.Sigmoid.CalculateAccuracy(options.TestSet, options.TestSetDesiredOutputs, inputLayerWeightsReshaped, hiddenLayerWeightsReshaped);
                        //    Log.Logger.AddEpoch(f2, accuracy);
                        //}
                        M = M - 1;
                        currentEpoch = currentEpoch + (options.Epochs < 0 ? 1 : 0);
                        d2 = innerGradient.DotProduct(s);
                        z3 = z3 - z2;
                    }
                    if (innerCost > cost + step * RHO * slope || d2 > -SIG * slope)
                    {
                        break;
                    }
                    else if (d2 > SIG * slope)
                    {
                        success = 1;
                        break;
                    }
                    else if (M == 0)
                    {
                        break;
                    }
                    double _A = 6 * (innerCost - f3) / z3 + 3 * (d2 + d3);
                    double _B = 3 * (f3 - innerCost) - z3 * (d3 + 2 * d2);
                    double _step = GetStep(step, d2, z3, limit, _A, _B);
                    f3 = innerCost;
                    d3 = d2;
                    z3 = -_step;
                    step = step + _step;
                    input = input.Add(s.Multiply(_step));
                    options.Weights = ReshapeWeights(input, options);
                    evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
                    innerCost = evaluate.Cost;
                    innerGradient = evaluate.Gradient;
                    //if (Log.Logger != null)
                    //{
                    //    Matrix<double> inputLayerWeightsReshaped;
                    //    Matrix<double> hiddenLayerWeightsReshaped;
                    //    Helpers.ReshapeMatrices(input, out inputLayerWeightsReshaped, out hiddenLayerWeightsReshaped, options.InputWeights.RowCount, options.InputWeights.ColumnCount, options.HiddenLayerWeights.RowCount, options.HiddenLayerWeights.ColumnCount);
                    //    double accuracy = HelperFunctions.Sigmoid.CalculateAccuracy(options.TestSet, options.TestSetDesiredOutputs, inputLayerWeightsReshaped, hiddenLayerWeightsReshaped);
                    //    Log.Logger.AddEpoch(f2, accuracy);
                    //}
                    M = M - 1;
                    currentEpoch = CountEpochs(options, currentEpoch);
                    d2 = innerGradient.DotProduct(s);
                }

                Vector<double> temporaryGradient = null;

                if (success == 1)
                {
                    cost = innerCost;
                    double numerator = (innerGradient.DotProduct(innerGradient) - gradient.DotProduct(innerGradient)) / gradient.DotProduct(gradient);
                    s = s.Multiply(numerator).Subtract(innerGradient);
                    temporaryGradient = gradient;
                    gradient = innerGradient;
                    innerGradient = temporaryGradient;
                    d2 = gradient.DotProduct(s);
                    if (d2 > 0)
                    {
                        s = gradient.Multiply(-1d);
                        d2 = s.Multiply(-1d).DotProduct(s);
                    }
                    step = step * System.Math.Min(RATIO, slope / (d2 - 2.2251e-308));
                    slope = d2;
                    lineSearchFailures = 0;
                }
                else
                {
                    input = inputSet;
                    cost = progressCost;
                    gradient = progressGradient;
                    if (LineSearchFailedOrRunOutOfEpochs(options, currentEpoch, lineSearchFailures))
                    {
                        break;
                    }
                    temporaryGradient = gradient;
                    gradient = innerGradient;
                    innerGradient = temporaryGradient;
                    s = gradient.Multiply(-1d);
                    slope = s.Multiply(-1d).DotProduct(s);
                    step = 1d / (1d - slope);
                    lineSearchFailures = 1;
                }
            }
            return input;
        }

        private static bool IsNotANumber(double z2)
        {
            return double.IsNaN(z2) || double.IsInfinity(z2);
        }

        private static bool LineSearchFailedOrRunOutOfEpochs(Options options, int currentEpoch, int lineSearchFailures)
        {
            return lineSearchFailures == 1 || currentEpoch > System.Math.Abs(options.Epochs);
        }

        private static double GetStep(double step, double d2, double z3, double limit, double _A, double _B)
        {
            double nextStep = -d2 * z3 * z3 / (_B + System.Math.Sqrt(_B * _B - _A * d2 * z3 * z3));
            if (ProblemOrAWrongSign(nextStep))
            {
                if (limit < -0.5)
                {
                    nextStep = ExtrapolateMaximumAmount(step);
                }
                else
                {
                    nextStep = Bisect(step, limit);
                }
            }
            else if ((limit > -0.5) && (nextStep + step > limit))
            {
                nextStep = Bisect(step, limit);
            }
            else if ((limit < -0.5) && (nextStep + step > step * EXT))
            {
                nextStep = ExtrapolateMaximumAmount(step);
            }
            else if (nextStep < -z3 * INT)
            {
                nextStep = -z3 * INT;
            }
            else if ((limit > -0.5) && (nextStep < (limit - step) * (1.0 - INT)))
            {
                nextStep = TooCloseToTheLimit(step, limit);
            }
            return nextStep;
        }

        private static bool EpochsNotFinished(Options options, int currentEpoch)
        {
            return currentEpoch < System.Math.Abs(options.Epochs);
        }

        private static int CountEpochs(Options options, int currentEpoch)
        {
            return currentEpoch + (options.Epochs < 0 ? 1 : 0);
        }

        private static double TooCloseToTheLimit(double step, double limit)
        {
            return (limit - step) * (1.0 - INT);
        }

        private static double Bisect(double step, double limit)
        {
            return (limit - step) / 2;
        }

        private static double ExtrapolateMaximumAmount(double step)
        {
            return step * (EXT - 1);
        }

        private static bool ProblemOrAWrongSign(double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value) || value < 0d;
        }

        private static int GetM(int epochs, int currentEpoch)
        {
            if (epochs > 0)
            {
                return MAX;
            }
            else
            {
                return System.Math.Min(MAX, -epochs - currentEpoch);
            }
        }

        private static System.Tuple<Matrix<double>, Matrix<double>> ReshapeWeights(Vector<double> unrolledWeights, Options options)
        {
            return new System.Tuple<Matrix<double>, Matrix<double>>(
                Matrix<double>.Build.DenseOfColumnMajor(options.Weights.Item1.RowCount, options.Weights.Item1.ColumnCount, unrolledWeights.SubVector(0, options.Weights.Item1.ColumnCount * options.Weights.Item1.RowCount)),
                Matrix<double>.Build.DenseOfColumnMajor(options.Weights.Item2.RowCount, options.Weights.Item2.ColumnCount, unrolledWeights.SubVector(options.Weights.Item1.ColumnCount * options.Weights.Item1.RowCount, options.Weights.Item2.ColumnCount * options.Weights.Item2.RowCount)));
        }
    }
}