using Microsoft.Win32;
using System;
using System.Windows;
using System.Windows.Controls;
using Logger;
using System.Collections.Generic;
using System.Windows.Media.Imaging;
using System.Windows.Media;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork;

namespace CharacterVisualizator
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        bool CanvasLoaded = false;
        private List<ANNLogItem> LogItems;
        char[] Letters = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z' };

        private void VisualizatorCanvas_Loaded(object sender, RoutedEventArgs e)
        {
            if (!CanvasLoaded)
            {
                var images = LoadImages();
                SetImages(images);
                CanvasLoaded = true;
            }
        }

        Image[] LoadImages()
        {
            var loader = new CharacterLoader();
            return loader.Load(@"DataSets\letter.data");
        }

        void SetImages(Image[] images)
        {
            int width = 30;
            int height = images.Length / width;
            int item = 0;

            for (int i = 0; i < width; ++i)
            {
                for (int j = 0; j < height; ++j)
                {
                    SetImage(i, j, images[item++]);
                }
            }
        }

        void SetImage(int i, int j, Image image)
        {
            Canvas.SetLeft(image, i * 8 + 5);
            Canvas.SetTop(image, j * 16 + 2);
            VisualizatorCanvas.Children.Add(image);
        }

        void btnSelectLog_Click(object sender, RoutedEventArgs e)
        {
            string filename = PickFile();
            if (String.IsNullOrEmpty(filename))
            {
                return;
            }
            LogItems = ANNLogger.ReadLogFile(filename);
        }

        string PickFile()
        {
            var filePicker = new OpenFileDialog();
            filePicker.InitialDirectory = AppDomain.CurrentDomain.BaseDirectory + @"Log\";
            filePicker.Filter = "Log (.json)|*.json";
            var result = filePicker.ShowDialog();
            if (result == true)
            {
                return filePicker.FileName;
            }
            return String.Empty;
        }

        void btnCheck_Click(object sender, RoutedEventArgs e)
        {
            if (LogItems == null) return;
            double[] pixels = GitPixels();
            var result = RecognizeChar(pixels);
            DiscoveredLetter.Text = result.ToString();
        }

        private double[] GitPixels()
        {
            var image = GetRawImage();
            var imageManipulation = new ImageManipulation((int)DrawingTable.ActualWidth, (int)DrawingTable.ActualHeight, image);
            LetterPreview.Children.Clear();
            LetterPreview.Children.Add(new Image { Source = imageManipulation.TransformSize() });
            var pixels = imageManipulation.GetBinaryPixels();
            return pixels;
        }

        char RecognizeChar(double[] pixels)
        {
            var costFunction = NeuralNetwork.CostFunctions.CostFunction.Build(CostFunctionTypes.Sigmoid);
            var log = LogItems.OrderBy(x => x.Epochs.OrderBy(y => y.Accuracy).Last().Accuracy).Last();
            var weights = Vector<double>.Build.DenseOfArray(log.Weights).ReshapeMatrices(log.NumberOfHiddenNeurons, 129, 26, log.NumberOfHiddenNeurons + 1);
            Vector<double> a1 = Vector<double>.Build.Random(1),
                z2=  Vector<double>.Build.Random(1),
                a2 = Vector<double>.Build.Random(1),
                a3 = Vector<double>.Build.Random(1);
            costFunction.FeedForward(weights.Item1, weights.Item2, Vector<double>.Build.DenseOfArray(pixels).ToRowMatrix(), 0, out a1, out z2, out a2, out a3);
            return Letters[a3.MaximumIndex()];
        }

        Image GetRawImage()
        {
            var picture = new RenderTargetBitmap((int)DrawingTable.ActualWidth, (int)DrawingTable.ActualHeight, 96, 96, PixelFormats.Default);
            picture.Render(DrawingTable);
            return new Image
            {
                Source = picture
            };
        }        

        void btnClear_Click(object sender, RoutedEventArgs e)
        {
            ClearDrawingTable();
            DiscoveredLetter.Text = "-";
        }

        private void ClearDrawingTable()
        {
            DrawingTable.Children.Clear();
            DrawingTable.Strokes.Clear();
            LetterPreview.Children.Clear();
        }
    }
}
