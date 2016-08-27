using ArtificialNeuralNetwork.Parsers;
using MathNet.Numerics.LinearAlgebra;
using System.Windows;

namespace ArtificialNeuralNetwork
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        Vector<double> mostAccurate;
        Vector<double> leastCost;

        private void TrainMany_Click(object sender, RoutedEventArgs e)
        {
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(@"DataSets\letter.data");
            double currentAccuracy = 0d;
            double currentCost = int.MaxValue;
            for (int epochs = 50; epochs < 100; epochs += 10)
            {
                for (int numberOfNeurons = 5; numberOfNeurons < 105; numberOfNeurons += 5)
                {
                    for (double lambda = 0.01d; lambda < 10.25d; lambda *= 2d)
                    {
                        var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(result.Item1, result.Item2, numberOfNeurons, epochs, lambda);
                        var learningResult = ann.Learn();
                        if (currentAccuracy < learningResult.Accuracy)
                        {
                            mostAccurate = learningResult.Gradient;
                            currentAccuracy = learningResult.Accuracy;
                        }
                        if (currentCost > learningResult.Cost)
                        {
                            leastCost = learningResult.Gradient;
                            currentCost = learningResult.Cost;
                        }
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
