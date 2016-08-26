using System.Collections.Generic;
using System.Runtime.Serialization;

namespace Logger
{
    [DataContract]
    public class ANNLogItem
    {
        public ANNLogItem()
        {
            Epochs = new List<ANNLogEpochItem>();
        }
        [DataMember]
        public int TrainingSetSize { get; set; }
        [DataMember]
        public int TestSetSize { get; set; }
        [DataMember]
        public double Lambda { get; set; }
        [DataMember]
        public string Function { get; set; }
        [DataMember]
        public int NumberOfEpochs { get; set; }
        [DataMember]
        public List<ANNLogEpochItem> Epochs { get; set; }
        [DataMember]
        public double TrainingTimeInMilliseconds { get; set; }
        [DataMember]
        public int NumberOfHiddenNeurons { get; set; }
    }
}
