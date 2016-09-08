using System;

namespace NeuralNetwork.CostFunctions
{
    class CostFunction
    {
        public static ICostFunction Build(CostFunctionTypes type)
        {
            switch (type)
            {
                case CostFunctionTypes.Sigmoid:
                    return new Sigmoid();
                default:
                    throw new NotImplementedException("This type of Cost Function is not implemented yet.");
            }
        }
    }
}
