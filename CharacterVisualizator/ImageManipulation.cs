using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace CharacterVisualizator
{
    class ImageManipulation
    {
        Image Image { get; }
        int Width { get; }
        int Height { get; }
        PixelFormat Format { get; }

        const int DesiredWidth = 8;
        const int DesiredHeight = 16;
        int TransformedPixelSize;

        public ImageManipulation(int controlWidth, int controlHeight, Image image)
        {
            Width = controlWidth;
            Height = controlHeight;
            Image = image;
        }

        public double[] GetBinaryPixels()
        {
            var resizedPixels = ResizeImage();
            return ConvertRawToBinaryPixels(resizedPixels);
        }

        byte[] ResizeImage()
        {
            TransformedBitmap bitmap = TransformSize();
            int stride = GetStride(bitmap);
            return GetPixels(bitmap, stride);
        }

        public TransformedBitmap TransformSize()
        {
            double scaleX = (double)DesiredWidth / Width,
                scaleY = (double)DesiredHeight / Height;
            return new TransformedBitmap(
                Image.Source as BitmapSource,
                new ScaleTransform(scaleX, scaleY));
        }

        int GetStride(TransformedBitmap bitmap)
        {
            var format = bitmap.Format;
            TransformedPixelSize = (format.BitsPerPixel + 7) / 8;
            return bitmap.PixelWidth * TransformedPixelSize;
        }

        byte[] GetPixels(TransformedBitmap bitmap, int stride)
        {
            var pixels = new byte[bitmap.PixelHeight * stride];
            bitmap.CopyPixels(pixels, stride, 0);
            return pixels;
        }

        double[] ConvertRawToBinaryPixels(byte[] pixelValues)
        {
            var binary = new double[DesiredHeight * DesiredWidth];
            for (int i = 0; i < pixelValues.Length; i += TransformedPixelSize)
            {
                int pixel = 0;
                for (int j = i; j < i + TransformedPixelSize; ++j)
                {
                    pixel += pixelValues[j];
                }
                pixel /= TransformedPixelSize;
                binary[i / TransformedPixelSize] = pixel <= 128 ? 1d : 0d;
            }
            return binary;
        }
    }
}
