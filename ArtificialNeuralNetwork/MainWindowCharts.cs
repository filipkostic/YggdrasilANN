﻿using Microsoft.Win32;
using System.Collections.Generic;
using System.Windows;
using ArtificialNeuralNetwork.Charting;
using Sparrow.Chart;
using System.Linq;
using System;
using System.Windows.Media;
using System.IO;

namespace ArtificialNeuralNetwork
{
    public partial class MainWindow
    {
        void Grid_Loaded(object sender, RoutedEventArgs e)
        {
            ViewModels = new Dictionary<int, ChartViewModel>();
            LogItems = new List<List<Logger.ANNLogItem>>();
            InitializeLegend(PlotMapCostEpoch, "Cost by epochs");
            InitializeLegend(PlotMapCostLambda, "Cost by lambda");
            InitializeLegend(PlotMapCostEpochIdeal, "Cost/Accuracy for 25 hidden neurons and lambda ≈ 1");
            InitializeLegend(PlotMapAccuracyEpoch, "Accuracy by epochs");
        }

        void InitializeLegend(SparrowChart chart, string text)
        {
            chart.Legend = new Legend();
            chart.Legend.Header = text;
            chart.Legend.LegendPosition = LegendPosition.Outside;
        }

        void LogPathButton_Click(object sender, RoutedEventArgs e)
        {
            string dirPath = PickLogFile();
            LogPath.Text = dirPath;
            if (dirPath == "")
                return;
            foreach (var file in Directory.GetFiles(dirPath))
            {
                LogItems.Add(Logger.ANNLogger.ReadLogFile(file));
            }
            var best = LogItems.Select(x => x.OrderBy(y => y.Epochs.Max(z => z.Accuracy)).Last());
            var item = best.OrderBy(x => x.Epochs.Max(y => y.Accuracy)).Last();
            BestLogItem = best.ToList();
            var epoch = item.Epochs.OrderBy(x => x.Accuracy).Last();
            MessageBox.Show("The best accuracy is achieved by the combination:\nHidden neurons:\t" + item.NumberOfHiddenNeurons
                + "\nLambda:\t" + item.Lambda
                + "\nEpoch:\t" + item.Epochs.IndexOf(epoch) + " of " + item.Epochs.Count
                + "\nAccuracy:\t" + epoch.Accuracy.ToString("0.00%"));
        }

        void AssignModelToCostEpochView(Logger.ANNLogItem item)
        {
            switch (item.NumberOfHiddenNeurons)
            {
                case 5:
                    Series01.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 10:
                    Series02.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 15:
                    Series03.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 20:
                    Series04.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 25:
                    Series05.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 30:
                    Series06.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 35:
                    Series07.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 40:
                    Series08.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 45:
                    Series09.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 50:
                    Series10.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 55:
                    Series11.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 60:
                    Series12.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 65:
                    Series13.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 70:
                    Series14.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 75:
                    Series15.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 80:
                    Series16.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 85:
                    Series17.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 90:
                    Series18.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 95:
                    Series19.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 100:
                    Series20.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
            }
        }

        void AssignModelToAccuracyEpochView(Logger.ANNLogItem item)
        {
            switch (item.NumberOfHiddenNeurons)
            {
                case 5:
                    AccuracySeries01.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 10:
                    AccuracySeries02.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 15:
                    AccuracySeries03.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 20:
                    AccuracySeries04.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 25:
                    AccuracySeries05.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 30:
                    AccuracySeries06.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 35:
                    AccuracySeries07.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 40:
                    AccuracySeries08.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 45:
                    AccuracySeries09.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 50:
                    AccuracySeries10.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 55:
                    AccuracySeries11.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 60:
                    AccuracySeries12.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 65:
                    AccuracySeries13.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 70:
                    AccuracySeries14.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 75:
                    AccuracySeries15.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 80:
                    AccuracySeries16.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 85:
                    AccuracySeries17.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 90:
                    AccuracySeries18.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 95:
                    AccuracySeries19.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
                case 100:
                    AccuracySeries20.DataContext = ViewModels[item.NumberOfHiddenNeurons];
                    break;
            }
        }

        static string PickLogFile()
        {
            var dialog = new OpenFileDialog();
            dialog.Filter = "Log file (.json)|*.json";
            dialog.RestoreDirectory = true;
            dialog.InitialDirectory = AppDomain.CurrentDomain.BaseDirectory;
            var result = dialog.ShowDialog();
            if (result == true)
            {
                int length = dialog.FileName.Length - (dialog.FileName.Length - dialog.FileName.LastIndexOf('\\'));
                return dialog.FileName.Substring(0, length);
            }
            return String.Empty;
        }

        public List<List<Logger.ANNLogItem>> LogItems { get; set; }
        List<Logger.ANNLogItem> BestLogItem { get; set; }
        public Dictionary<int, ChartViewModel> ViewModels;

        void CostEpochs_Checked(object sender, RoutedEventArgs e)
        {
            if (LogItems == null) return;
            foreach (var item in ViewModels)
            {
                item.Value.Collection.Clear();
            }
            PlotMapCostEpoch.XAxis.ShowCrossLines = true;
            PlotMapCostEpoch.XAxis.Header = "Epochs";
            PlotMapCostEpoch.YAxis.ShowCrossLines = true;
            PlotMapCostEpoch.YAxis.Header = "Cost";
            var filter = BestLogItem.Where(x => x.Lambda < 2 && x.Lambda >= 1);
            foreach (var item in filter)
            {
                ViewModels.Add(item.NumberOfHiddenNeurons, new ChartViewModel());

                AssignModelToCostEpochView(item);
                for (int epoch = 0; epoch < item.Epochs.Count; ++epoch)
                {
                    ViewModels[item.NumberOfHiddenNeurons].Collection.Add(
                        new Charting.Point(epoch + 1, item.Epochs[epoch].Cost, String.Format("{0} hidden neurons", item.NumberOfHiddenNeurons)));
                }
            }
            ShowChart(PlotMapCostEpoch);
        }

        void CostLambda_Checked(object sender, RoutedEventArgs e)
        {
            if (LogItems == null) return;
            PlotMapCostLambda.XAxis.ShowCrossLines = true;
            PlotMapCostLambda.XAxis.Header = "Lambda";
            PlotMapCostLambda.YAxis.ShowCrossLines = true;
            PlotMapCostLambda.YAxis.Header = "Cost/Accuracy";
            var filter = BestLogItem
                .Where(x => x.NumberOfHiddenNeurons == 25)
                .OrderBy(x => x.Lambda)
                .ToList();
            var minCost = new ChartViewModel();
            var minCostAccuracy = new ChartViewModel();
            var maxAccuracy = new ChartViewModel();
            var maxAccuracyCost = new ChartViewModel();
            LambdaSeriesMinCost.DataContext = minCost;
            LambdaSeriesMinCostAccuracy.DataContext = minCostAccuracy;
            LambdaSeriesMinCostAccuracy.Fill.Opacity = 0.8d;
            LambdaSeriesMaxAccuracy.DataContext = maxAccuracy;
            var b1 = new SolidColorBrush(Colors.Red);
            b1.Opacity = 0.6d;
            LambdaSeriesMaxAccuracy.Fill = b1;
            LambdaSeriesMaxAccuracyCost.DataContext = maxAccuracyCost;
            var b = new SolidColorBrush(Colors.Blue);
            b.Opacity = 0.3d;
            LambdaSeriesMaxAccuracyCost.Fill = b;

            for (int j = 0; j < filter.Count; ++j)
            {
                var minCostEpoch = filter[j].Epochs.OrderBy(x => x.Cost).First();
                var maxAccuracyEpoch = filter[j].Epochs.OrderBy(x => x.Accuracy).Last();
                minCost.Collection.Add(new Charting.Point(filter[j].Lambda, minCostEpoch.Cost, ""));
                minCostAccuracy.Collection.Add(new Charting.Point(filter[j].Lambda, minCostEpoch.Accuracy, ""));
                maxAccuracy.Collection.Add(new Charting.Point(filter[j].Lambda, maxAccuracyEpoch.Accuracy, ""));
                maxAccuracyCost.Collection.Add(new Charting.Point(filter[j].Lambda, maxAccuracyEpoch.Cost, ""));
            }
            ShowChart(PlotMapCostLambda);
        }

        void IdealExample_Checked(object sender, RoutedEventArgs e)
        {
            if (LogItems == null) return;
            PlotMapCostEpochIdeal.XAxis.ShowCrossLines = true;
            PlotMapCostEpochIdeal.XAxis.Header = "Epoch";
            PlotMapCostEpochIdeal.YAxis.ShowCrossLines = true;
            PlotMapCostEpochIdeal.YAxis.Header = "Cost/Accuracy";
            var filter = BestLogItem/*.Where(x => x.Lambda < 2 && x.Lambda >= 1 && x.NumberOfHiddenNeurons == 25)*/.FirstOrDefault();

            var cost = new ChartViewModel();
            var accuracy = new ChartViewModel();

            for (int i = 0; i < filter.Epochs.Count; ++i)
            {
                cost.Collection.Add(new Charting.Point(i + 1, filter.Epochs[i].Cost, ""));
                accuracy.Collection.Add(new Charting.Point(i + 1, filter.Epochs[i].Accuracy, ""));
            }

            IdealExampleCostSeries.DataContext = cost;
            IdealExampleAccuracySeries.DataContext = accuracy;
            IdealExampleAccuracySeries.Fill.Opacity = 0.4d;
            ShowChart(PlotMapCostEpochIdeal);
        }

        void AccuracyEpochs_Checked(object sender, RoutedEventArgs e)
        {
            if (LogItems == null) return;
            foreach (var item in ViewModels)
            {
                item.Value.Collection.Clear();
            }
            PlotMapAccuracyEpoch.XAxis.ShowCrossLines = true;
            PlotMapAccuracyEpoch.XAxis.Header = "Epochs";
            PlotMapAccuracyEpoch.YAxis.ShowCrossLines = true;
            PlotMapAccuracyEpoch.YAxis.Header = "Accuracy";
            var filter = BestLogItem;//.Where(x => x.Lambda < 2 && x.Lambda >= 1);
            foreach (var item in filter)
            {
                ViewModels.Add(item.NumberOfHiddenNeurons, new ChartViewModel());

                AssignModelToAccuracyEpochView(item);
                for (int epoch = 0; epoch < item.Epochs.Count; ++epoch)
                {
                    ViewModels[item.NumberOfHiddenNeurons].Collection.Add(
                        new Charting.Point(epoch + 1, item.Epochs[epoch].Accuracy, String.Format("{0} hidden neurons", item.NumberOfHiddenNeurons)));
                }
            }
            ShowChart(PlotMapAccuracyEpoch);
        }

        void ShowChart(SparrowChart chart)
        {
            PlotMapCostEpoch.Visibility = ToggleChartVisibility(chart, PlotMapCostEpoch);
            PlotMapCostLambda.Visibility = ToggleChartVisibility(chart, PlotMapCostLambda);
            PlotMapCostEpochIdeal.Visibility = ToggleChartVisibility(chart, PlotMapCostEpochIdeal);
            PlotMapAccuracyEpoch.Visibility = ToggleChartVisibility(chart, PlotMapAccuracyEpoch);
        }

        Visibility ToggleChartVisibility(SparrowChart chart, SparrowChart toggle)
        {
            return chart == toggle ? Visibility.Visible : Visibility.Hidden;
        }
    }
}
