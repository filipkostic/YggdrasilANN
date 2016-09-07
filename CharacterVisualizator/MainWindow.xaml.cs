using System.Windows;
using System.Windows.Controls;

namespace CharacterVisualizator
{
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        bool CanvasLoaded = false;

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
    }
}
