using MathNet.Numerics.LinearAlgebra;
using System;
using System.Threading;
using System.Windows;
using ContentParsers;

namespace ArtificialNeuralNetwork
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        void TrainMany_Click(object sender, RoutedEventArgs e)
        {
            ThreadPool.SetMaxThreads(20, 25);
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(@"DataSets\letter.data");
            for (int epochs = 30; epochs <= 100; epochs += 10)
            {
                for (int numberOfNeurons = 15; numberOfNeurons <= 100 ; numberOfNeurons += 5)
                {
                    for (double lambda = 0.01d; lambda <= 10.25d; lambda *= 2d)
                    {
                        ThreadPool.QueueUserWorkItem(new WaitCallback(x => ANN_Worker(lambda, numberOfNeurons, epochs, result)));
                    }
                }
            }
        }

        void ANN_Worker(double lambda, int hln, int epochs, Tuple<Matrix<double>, Matrix<double>> set)
        {
            var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(set.Item1, set.Item2, hln, epochs, lambda);
            var learningResult = ann.Learn();
        }

        void TrainOne_Click(object sender, RoutedEventArgs e)
        {
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(@"DataSets\letter.data");
            var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(result.Item1, result.Item2, 50, 100, 0.16d);
            var learningResult = ann.Learn();
        }
    }
}
