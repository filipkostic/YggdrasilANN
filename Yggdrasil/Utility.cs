using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class Utility
    {
        internal static int[] UniqueRandomArray(int max, int numberOfItems)
        {
            Random generator = new Random();
            List<int> numbers = new List<int>();
            while (numbers.Count < numberOfItems)
            {
                int number = generator.Next(max);
                if (!numbers.Contains(number))
                {
                    numbers.Add(number);
                }
            }
            return numbers.ToArray();
        }
    }
}
