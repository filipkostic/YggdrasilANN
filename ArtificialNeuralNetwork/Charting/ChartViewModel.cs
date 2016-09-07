using System.Collections.ObjectModel;

namespace ArtificialNeuralNetwork.Charting
{
    public class ChartViewModel
    {
        public ObservableCollection<Point> Collection { get; set; }
        public ChartViewModel()
        {
            Collection = new ObservableCollection<Point>();
        }
    }
}
