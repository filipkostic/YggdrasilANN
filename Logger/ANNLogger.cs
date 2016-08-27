using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Json;

namespace Logger
{
    public class ANNLogger
    {
        public ANNLogger(string logFilePath)
        {
            LogItems = new List<ANNLogItem>();
            LoggingInProgress = false;
            LogFilePath = logFilePath;
        }

        public void Start(int trainingSize, int testingSize, int epochs, double lambda, int hiddenNeurons, string function)
        {
            if (LoggingInProgress)
            {
                throw new Exception("There is already logging in progress.");
            }
            CurrentLogItem = new ANNLogItem
            {
                TrainingSetSize = trainingSize,
                TestSetSize = testingSize,
                NumberOfEpochs = epochs,
                Lambda = lambda,
                NumberOfHiddenNeurons = hiddenNeurons,
                Function = function
            };
            StartTime = DateTime.Now;
            LoggingInProgress = true;
        }

        public List<ANNLogItem> LogItems { get; }

        public void AddEpoch(int epoch, double cost, double accuracy)
        {
            var currentEpoch = CurrentLogItem.Epochs.FirstOrDefault(x => x.Epoch == epoch);
            if (currentEpoch == null)
            {
                CurrentLogItem.Epochs.Add(new ANNLogEpochItem { Epoch = epoch, Cost = cost, Accuracy = accuracy });
            }
            else
            {
                currentEpoch.Cost = cost;
                currentEpoch.Accuracy = accuracy;
            }
        }

        ANNLogItem CurrentLogItem;
        DateTime StartTime;
        DateTime EndTime;
        string LogFilePath;
        public bool LoggingInProgress { get; private set; }

        public void Finish()
        {
            if (!LoggingInProgress)
            {
                throw new Exception("There is no logging in progress.");
            }
            LoggingInProgress = false;
            EndTime = DateTime.Now;
            CurrentLogItem.TrainingTimeInMilliseconds = (EndTime - StartTime).TotalMilliseconds;
            LogItems.Add(CurrentLogItem);
            CurrentLogItem = null;
            WriteToLogFile();
        }

        public void WriteToLogFile()
        {
            string logDirectory = "Log";
            if (!Directory.Exists(logDirectory))
            {
                Directory.CreateDirectory(logDirectory);
            }
            using (var stream = new FileStream(logDirectory + "\\" + LogFilePath, FileMode.Create))
            {
                var serializer = new DataContractJsonSerializer(typeof(List<ANNLogItem>));
                serializer.WriteObject(stream, LogItems);
            }
        }

        public static List<ANNLogItem> ReadLogFile(string logFilePath)
        {
            List<ANNLogItem> log;
            using (var stream = new FileStream(logFilePath, FileMode.Open))
            {
                var serializer = new DataContractJsonSerializer(typeof(List<ANNLogItem>));
                log = (List<ANNLogItem>)serializer.ReadObject(stream);
            }
            return log;
        }
    }
}
