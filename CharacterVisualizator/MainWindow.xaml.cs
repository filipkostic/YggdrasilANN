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
            tbxLogFilePath.Text = filename;
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
            var pixels = imageManipulation.GetBinaryPixels();
            LetterPreview.Children.Clear();
            CharacterLoader loader = new CharacterLoader();
            LetterPreview.Children.Add(loader.Load(pixels));
            return pixels;
        }

        char RecognizeChar(double[] pixels)
        {
            var costFunction = NeuralNetwork.CostFunctions.CostFunction.Build(CostFunctionTypes.Sigmoid);
            var log = LogItems.OrderBy(x => x.Epochs.OrderBy(y => y.Accuracy).Last().Accuracy).Last();

            var outputs = costFunction.FeedForward(log.Weights, log.NumberOfHiddenNeurons, 26, pixels);
            return Letters[IndexOfMax(outputs)];
        }

        int IndexOfMax(double[] outputs)
        {
            double max = outputs.Max();
            for (int i = 0; i < outputs.Length; ++i)
            {
                if (outputs[i] == max)
                {
                    return i;
                }
            }
            return -1;
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

        void ClearDrawingTable()
        {
            DrawingTable.Children.Clear();
            DrawingTable.Strokes.Clear();
            LetterPreview.Children.Clear();
        }
    }
}
