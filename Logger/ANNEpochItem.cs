using System.Runtime.Serialization;

namespace Logger
{
    [DataContract]
    public class ANNLogEpochItem
    {
        [DataMember]
        public double Cost { get; set; }
        [DataMember]
        public double Accuracy { get; set; }
    }
}
