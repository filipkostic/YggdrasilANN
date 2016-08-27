using MathNet.Numerics.LinearAlgebra;
using System;

namespace NeuralNetwork
{
    class Fmincg
    {
        Fmincg()
        {

        }

        public static Fmincg Build
        {
            get
            {
                return new Fmincg();
            }
        }

        public delegate ICostGradientResult CostFunction(Matrix<double> inputWeights, Matrix<double> hiddenLayerWeights, Matrix<double> inputs, Matrix<double> desiredOutputs, double lambda);
        public delegate void LearningEventDelegate(int epoch, Options options, Vector<double> data, double cost);
        public event LearningEventDelegate LearningEvent;

        const double EXT = 3.0;
        const double RealMin = 2.2251e-308;
        const double RHO = 0.01;
        const double SIG = 0.5;
        const double INT = 0.1;
        const int MAX = 20;
        const int RATIO = 100;

        public Vector<double> MinimizeFunction(CostFunction costFunction, Vector<double> theta, Options options)
        {
            return Minimize(costFunction, theta, options);
        }

        Vector<double> Minimize(CostFunction costFunction, Vector<double> theta, Options options)
        {
            Vector<double> input = theta;
            int currentEpoch = 0;
            int red = 1;
            int lineSearchFailures = 0;
            ICostGradientResult evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
            double cost = evaluate.Cost;
            Vector<double> gradient = evaluate.Gradient;
            LearningEvent(currentEpoch, options, input, cost);
            
            currentEpoch = currentEpoch + (options.Epochs < 0 ? 1 : 0);
            Vector<double> inverseGradient = gradient.Multiply(-1d);

            double slope = inverseGradient.Multiply(-1d).DotProduct(inverseGradient);
            double step = red / (1.0 - slope);

            while (EpochsNotFinished(options, currentEpoch))
            {
                currentEpoch = currentEpoch + (options.Epochs > 0 ? 1 : 0);
                Vector<double> inputSet = Vector<double>.Build.DenseOfVector(input);
                double progressCost = cost;
                Vector<double> progressGradient = Vector<double>.Build.DenseOfVector(gradient);
                input = input.Add(inverseGradient.Multiply(step));
                options.Weights = input.ReshapeMatrices(options.Weights.Item1.RowCount, options.Weights.Item1.ColumnCount, options.Weights.Item2.RowCount, options.Weights.Item2.ColumnCount);
                evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
                double epochCost = evaluate.Cost;
                Vector<double> innerGradient = evaluate.Gradient;
                LearningEvent(currentEpoch, options, input, epochCost);

                currentEpoch = currentEpoch + (options.Epochs < 0 ? 1 : 0);
                double epochSlope = innerGradient.DotProduct(inverseGradient);
                double innerEpochCost = cost;
                double innerEpochSlope = slope;
                double stepBack = -step;
                int M = GetM(options.Epochs, currentEpoch);
                int success = 0;
                double limit = -1;

                while (true)
                {
                    while (((epochCost > cost + step * RHO * slope) | (epochSlope > -SIG * slope)) && (M > 0))
                    {
                        limit = step;
                        double innerStep = GetInnerStep(cost, epochCost, epochSlope, innerEpochCost, innerEpochSlope, stepBack);
                        step = step + innerStep;
                        input = input.Add(inverseGradient.Multiply(innerStep));
                        options.Weights = input.ReshapeMatrices(options.Weights.Item1.RowCount, options.Weights.Item1.ColumnCount, options.Weights.Item2.RowCount, options.Weights.Item2.ColumnCount);
                        evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
                        epochCost = evaluate.Cost;
                        innerGradient = evaluate.Gradient;
                        LearningEvent(currentEpoch, options, input, epochCost);
                        M = M - 1;
                        currentEpoch = currentEpoch + (options.Epochs < 0 ? 1 : 0);
                        epochSlope = innerGradient.DotProduct(inverseGradient);
                        stepBack = stepBack - innerStep;
                    }
                    if (epochCost > cost + step * RHO * slope || epochSlope > -SIG * slope)
                    {
                        break;
                    }
                    else if (IsSuccessfullEpoch(slope, epochSlope))
                    {
                        success = 1;
                        break;
                    }
                    else if (M == 0)
                    {
                        break;
                    }

                    double epochStep = GetStep(step, epochSlope, innerEpochSlope, stepBack, limit, epochCost, innerEpochCost);
                    innerEpochCost = epochCost;
                    innerEpochSlope = epochSlope;
                    stepBack = -epochStep;
                    step = step + epochStep;
                    input = input.Add(inverseGradient.Multiply(epochStep));
                    options.Weights = input.ReshapeMatrices(options.Weights.Item1.RowCount, options.Weights.Item1.ColumnCount, options.Weights.Item2.RowCount, options.Weights.Item2.ColumnCount);
                    evaluate = costFunction(options.Weights.Item1, options.Weights.Item2, options.Training.Set, options.Training.Desired, options.Lambda);
                    epochCost = evaluate.Cost;
                    innerGradient = evaluate.Gradient;
                    LearningEvent(currentEpoch, options, input, epochCost);
                    M = M - 1;
                    currentEpoch = CountEpochs(options, currentEpoch);
                    epochSlope = innerGradient.DotProduct(inverseGradient);
                }

                if (success == 1)
                {
                    cost = epochCost;
                    double numerator = (innerGradient.DotProduct(innerGradient) - gradient.DotProduct(innerGradient)) / gradient.DotProduct(gradient);
                    inverseGradient = inverseGradient.Multiply(numerator).Subtract(innerGradient);
                    Vector<double> temporaryGradient = temporaryGradient = gradient;
                    gradient = innerGradient;
                    innerGradient = temporaryGradient;
                    epochSlope = gradient.DotProduct(inverseGradient);
                    if (epochSlope > 0)
                    {
                        inverseGradient = gradient.Multiply(-1d);
                        epochSlope = inverseGradient.Multiply(-1d).DotProduct(inverseGradient);
                    }
                    step = step * Math.Min(RATIO, slope / (epochSlope - RealMin));
                    slope = epochSlope;
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
                    Vector<double> temporaryGradient = gradient;
                    gradient = innerGradient;
                    innerGradient = temporaryGradient;
                    inverseGradient = gradient.Multiply(-1d);
                    slope = inverseGradient.Multiply(-1d).DotProduct(inverseGradient);
                    step = 1d / (1d - slope);
                    lineSearchFailures = 1;
                }
            }
            return input;
        }

        bool EpochsNotFinished(Options options, int currentEpoch)
        {
            return currentEpoch < Math.Abs(options.Epochs);
        }

        int GetM(int epochs, int currentEpoch)
        {
            if (epochs > 0)
            {
                return MAX;
            }
            else
            {
                return Math.Min(MAX, -epochs - currentEpoch);
            }
        }

        bool IsSuccessfullEpoch(double slope, double epochSlope)
        {
            return epochSlope > SIG * slope;
        }

        double GetInnerStep(double cost, double epochCost, double slope, double innerEpochCost, double innerSlope, double stepBack)
        {
            double innerStep = 0.0d;
            if (epochCost > cost)
            {
                innerStep = stepBack - (0.5 * innerSlope * Math.Pow(stepBack, 2)) / (innerSlope * stepBack + epochCost - innerEpochCost);
            }
            else
            {
                double A = 6 * (epochCost - innerEpochCost) / stepBack + 3 * (slope + innerSlope);
                double B = 3 * (innerEpochCost - epochCost) - stepBack * (innerSlope + 2 * slope);
                innerStep = (Math.Sqrt(B * B - A * slope * Math.Pow(stepBack, 2)) - B) / A;
            }
            if (IsNotANumber(innerStep))
            {
                innerStep = stepBack / 2.0d;
            }
            innerStep = Math.Max(Math.Min(innerStep, INT * stepBack), (1 - INT) * stepBack);
            return innerStep;
        }

        bool IsNotANumber(double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value);
        }

        bool LineSearchFailedOrRunOutOfEpochs(Options options, int currentEpoch, int lineSearchFailures)
        {
            return lineSearchFailures == 1 || currentEpoch > Math.Abs(options.Epochs);
        }

        double GetStep(double step, double epochSlope, double innerEpochSlope, double stepBack, double limit, double epochCost, double innerEpochCost)
        {
            double _A = 6 * (epochCost - innerEpochCost) / stepBack + 3 * (epochSlope + innerEpochSlope);
            double _B = 3 * (innerEpochCost - epochCost) - stepBack * (innerEpochSlope + 2 * epochSlope);
            double nextStep = -epochSlope * Math.Pow(stepBack, 2) / (_B + Math.Sqrt(Math.Pow(_B, 2) - _A * epochSlope * Math.Pow(stepBack, 2)));
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
            else if (nextStep < -stepBack * INT)
            {
                nextStep = -stepBack * INT;
            }
            else if ((limit > -0.5) && (nextStep < (limit - step) * (1.0 - INT)))
            {
                nextStep = TooCloseToTheLimit(step, limit);
            }
            return nextStep;
        }

        bool ProblemOrAWrongSign(double value)
        {
            return double.IsNaN(value) || double.IsInfinity(value) || value < 0d;
        }

        double ExtrapolateMaximumAmount(double step)
        {
            return step * (EXT - 1);
        }

        double Bisect(double step, double limit)
        {
            return (limit - step) / 2;
        }

        double TooCloseToTheLimit(double step, double limit)
        {
            return (limit - step) * (1.0 - INT);
        }

        int CountEpochs(Options options, int currentEpoch)
        {
            return currentEpoch + (options.Epochs < 0 ? 1 : 0);
        }
    }
}