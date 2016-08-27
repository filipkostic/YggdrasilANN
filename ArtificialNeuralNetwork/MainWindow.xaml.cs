using ArtificialNeuralNetwork.Parsers;
using MathNet.Numerics.LinearAlgebra;
using System.Threading;
using System.Windows;

namespace ArtificialNeuralNetwork
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void TrainMany_Click(object sender, RoutedEventArgs e)
        {
            ThreadPool.SetMaxThreads(20, 1);
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(@"DataSets\letter.data");
            for (int epochs = 50; epochs < 100; epochs += 10)
            {
                for (int numberOfNeurons = 5; numberOfNeurons < 105; numberOfNeurons += 5)
                {
                    for (double lambda = 0.01d; lambda < 10.25d; lambda *= 2d)
                    {
                        ThreadPool.QueueUserWorkItem(new WaitCallback((x) =>
                        {
                            var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(result.Item1, result.Item2, numberOfNeurons, epochs, lambda);
                            var learningResult = ann.Learn();
                        }));
                    }
                }
            }
        }

        private void TrainOne_Click(object sender, RoutedEventArgs e)
        {
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(@"DataSets\letter.data");
            var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(result.Item1, result.Item2);
            var learningResult = ann.Learn();
        }
    }
}
