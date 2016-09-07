using System.Windows;

namespace ArtificialNeuralNetwork.Charting
{
    public class Point : DependencyObject
    {
        public double X { get; set; }
        public double Y { get; set; }

        public string Label
        {
            get { return (string)GetValue(LabelProperty); }
            set { SetValue(LabelProperty, value); }
        }
        
        public static readonly DependencyProperty LabelProperty =
            DependencyProperty.Register("Label", typeof(string), typeof(Point), new PropertyMetadata(""));

        public Point(double x, double y, string label)
        {
            X = x;
            Y = y;
            Label = label;
        }
    }
}
