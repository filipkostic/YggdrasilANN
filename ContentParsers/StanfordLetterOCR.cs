using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ContentParsers
{
    public class StanfordLetterOCR : IParser
    {
        readonly char[] SplitCharacters = { '\t' };
        readonly int[] UnnecessaryColumns = { 0, 1, 2, 3, 4, 5 };
        const int CharacterIndex = 1;

        public Tuple<Matrix<double>, Matrix<double>> Read(string path)
        {
            using (StreamReader stream = File.OpenText(path))
            {
                return new Tuple<Matrix<double>, Matrix<double>>(
                    ReadDataMatrix(stream),
                    ReadDesiredMatrix(stream));
            }
        }


        private Matrix<double> ReadDataMatrix(StreamReader stream)
        {
            Matrix<double> data = InitializeDataMatix(stream);
            string[] line;
            for (int row = 0; row < data.RowCount; ++row)
            {
                line = GetRowItems(stream);
                for (int column = UnnecessaryColumns.Length; column < line.Length; ++column)
                {
                    data[row, column - UnnecessaryColumns.Length] = Double.Parse(line[column]);
                }
            }
            ResetStream(stream);
            return data;
        }

        Matrix<double> InitializeDataMatix(StreamReader stream)
        {
            int columns = GetNumberOfColumns(stream);
            int rows = GetNumberOfRows(stream);
            return Matrix<double>.Build.Dense(rows, columns);
        }

        int GetNumberOfColumns(StreamReader stream)
        {
            int columns = GetRowItems(stream).Length - UnnecessaryColumns.Length;
            ResetStream(stream);
            return columns;
        }

        private string[] GetRowItems(StreamReader stream)
        {
            return stream.ReadLine().Split(SplitCharacters, StringSplitOptions.RemoveEmptyEntries);
        }

        int GetNumberOfRows(StreamReader stream)
        {
            int i = 0;
            string buffer;
            while (!stream.EndOfStream)
            {
                buffer = stream.ReadLine();
                i++;
            }
            ResetStream(stream);
            return i;
        }

        void ResetStream(StreamReader stream)
        {
            stream.BaseStream.Position = 0;
            stream.DiscardBufferedData();
        }

        private Matrix<double> ReadDesiredMatrix(StreamReader stream)
        {
            int rows = GetNumberOfRows(stream);
            Dictionary<char, List<int>> charactersAndIndexes = ReadCharacterRows(stream, rows);
            var data = Matrix<double>.Build.Dense(rows, charactersAndIndexes.Count);
            int column = 0;
            foreach (var character in charactersAndIndexes)
            {
                char ch = character.Key;
                for (int row = 0; row < rows; ++row)
                {
                    data[row, column] = character.Value.Contains(row) ? 1 : 0;
                }
                column++;
            }
            return data;
        }

        private Dictionary<char, List<int>> ReadCharacterRows(StreamReader stream, int rows)
        {
            var charactersAndIndexes = new Dictionary<char, List<int>>();
            for (int row = 0; row < rows; ++row)
            {
                char letter = Char.Parse(GetRowItems(stream)[CharacterIndex]);
                if (!charactersAndIndexes.ContainsKey(letter))
                {
                    charactersAndIndexes[letter] = new List<int>();
                }
                charactersAndIndexes[letter].Add(row);
            }
            ResetStream(stream);
            charactersAndIndexes.OrderBy(x => x.Key);
            return charactersAndIndexes;
        }
    }
}
