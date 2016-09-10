using System.Linq;
using ContentParsers;
using System.Windows.Media.Imaging;
using System.Windows.Controls;
using System.Collections.Generic;
using System.Windows.Media;
using MathNet.Numerics.LinearAlgebra;

namespace CharacterVisualizator
{
    class CharacterLoader
    {
        public Image[] Load(string path)
        {
            var trainingSet = LoadSet(path);

            int width = 8, height = 16;
            var format = PixelFormats.Gray8;
            int bytesPerPixel = (format.BitsPerPixel + 7) / 8;
            int stride = bytesPerPixel * width;

            return GetImages(trainingSet, width, height, format, stride);
        }

        public Image Load(double[] pixels)
        {
            int width = 8, height = 16;
            var format = PixelFormats.Gray8;
            int bytesPerPixel = (format.BitsPerPixel + 7) / 8;
            int stride = bytesPerPixel * width;
            byte[] pixl = GetPixels(Vector<double>.Build.DenseOfArray(pixels).ToRowMatrix(), 0);
            return CreateImage(width, height, format, pixl, stride);
        }

        Matrix<double> LoadSet(string path)
        {
            IParser parser = new StanfordLetterOCR();
            var result = parser.Read(path);
            return result.Item1;
        }

        private Image[] GetImages(Matrix<double> set, int width, int height, PixelFormat format, int stride)
        {
            var images = new List<Image>();
            for (int index = 0; index < 200; ++index)
            {
                byte[] pixels = GetPixels(set, index);
                images.Add(CreateImage(width, height, format, pixels, stride));
            }
            return images.ToArray();
        }

        byte[] GetPixels(Matrix<double> set, int index)
        {
            return set
                .Row(index)
                .Select(x => (byte)(x == 1 ? 0x0 : 0xFFF))
                .ToArray();
        }

        Image CreateImage(int width, int height, PixelFormat format, byte[] pixels, int stride)
        {
            var bitmap = BitmapSource.Create(width, height, 96, 96, format, null, pixels, stride);
            bitmap.Freeze();
            return new Image
            {
                Source = bitmap
            };
        }
    }
}
