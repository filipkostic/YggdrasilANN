using ArtificialNeuralNetwork.Parsers;
using System.Windows;

namespace ArtificialNeuralNetwork
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(@"C:\Users\f.kostic\Downloads\letter.data");
            var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(result.Item1, result.Item2);
            var learningResult = ann.Learn();
        }
    }
}
