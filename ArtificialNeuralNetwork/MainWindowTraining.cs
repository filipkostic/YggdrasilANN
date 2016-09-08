using ContentParsers;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.Win32;
using System;
using System.Threading;
using System.Windows;

namespace ArtificialNeuralNetwork
{
    public partial class MainWindow
    {
        void TrainMany_Click(object sender, RoutedEventArgs e)
        {
            int fromEpochs = 0, toEpochs = 0, fromHln = 0, toHln = 0;
            double fromLambda = 0d, toLambda = 0d;
            bool success = Int32.TryParse(tbxFromEpochs.Text.Trim(), out fromEpochs)
                && Int32.TryParse(tbxToEpochs.Text.Trim(), out toEpochs)
                && Int32.TryParse(tbxFromNeurons.Text.Trim(), out fromHln)
                && Int32.TryParse(tbxToNeurons.Text.Trim(), out toHln)
                && Double.TryParse(tbxFromLambda.Text.Trim(), out fromLambda)
                && Double.TryParse(tbxToLambda.Text.Trim(), out toLambda);
            if (!success)
            {
                return;
            }
            var result = LoadData(@"DataSets\letter.data");
            for (int epochs = fromEpochs; epochs <= toEpochs; epochs += 10)
            {
                for (int numberOfNeurons = fromHln; numberOfNeurons <= toHln; numberOfNeurons += 5)
                {
                    for (double lambda = fromLambda; lambda <= toLambda; lambda *= 2d)
                    {
                        ThreadPool.QueueUserWorkItem(new WaitCallback(x => ANN_Worker(lambda, numberOfNeurons, epochs, result)));
                    }
                }
            }
        }

        void ANN_Worker(double lambda, int hln, int epochs, Tuple<Matrix<double>, Matrix<double>> set)
        {
            var ann = NeuralNetwork.ArtificialNeuralNetwork.Build(set.Item1, set.Item2, hln, epochs, lambda, NeuralNetwork.CostFunctionTypes.Sigmoid);
            var learningResult = ann.Learn();
        }

        void TrainOne_Click(object sender, RoutedEventArgs e)
        {
            int epochs = 0, hln = 0;
            double lambda = 0d;
            bool success = Int32.TryParse(tbxEpochs.Text.Trim(), out epochs)
                && Int32.TryParse(tbxNeurons.Text.Trim(), out hln)
                && Double.TryParse(tbxLambda.Text.Trim(), out lambda);
            if (!success)
            {
                return;
            }
            var result = LoadData(@"DataSets\letter.data");
            ANN_Worker(lambda, hln, epochs, result);
        }

        Tuple<Matrix<double>, Matrix<double>> LoadData(string path)
        {
            IParser parser = new StanfordLetterOCR();
            return parser.Read(@"DataSets\letter.data");
        }

        void btnPickTrainingDataMany_Click(object sender, RoutedEventArgs e)
        {
            tbxTrainingPathMany.Text = PickFile();
        }

        void btnPickTrainingData_Click(object sender, RoutedEventArgs e)
        {
            tbxTrainingPath.Text = PickFile();
        }

        string PickFile()
        {
            var filePicker = new OpenFileDialog();
            filePicker.InitialDirectory = AppDomain.CurrentDomain.BaseDirectory + @"DataSets\";
            filePicker.Filter = "Data (.data)|*.data|Text (.txt)|*.txt|Comma separated (.csv)|*.csv";
            var result = filePicker.ShowDialog();
            if (result == true)
            {
                return filePicker.FileName;
            }
            return String.Empty;
        }
    }
}
