using Microsoft.Win32;
using System;
using System.Windows;
using System.Windows.Controls;
using Logger;
using System.Collections.Generic;

namespace CharacterVisualizator
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        bool CanvasLoaded = false;
        private List<ANNLogItem> LogItem;

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
            LogItem = ANNLogger.ReadLogFile(filename);
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

        }
    }
}
