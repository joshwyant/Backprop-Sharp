using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;

namespace MNIST
{
    public class MNIST
    {
        public (float[] image, float[] value)[] TrainingData { get; }
        public (float[] image, float[] value)[] HoldoutData { get; }
        public (float[] image, float[] value)[] TestData { get; }
        
        public MNIST()
        {
            if (!Directory.Exists("data"))
            {
                Directory.CreateDirectory("data");
            }
            
            Download("train-images-idx3-ubyte.gz");
            Download("train-labels-idx1-ubyte.gz");
            Download("t10k-images-idx3-ubyte.gz");
            Download("t10k-labels-idx1-ubyte.gz");

            using (var labels = File.OpenRead("./data/train-labels-idx1-ubyte"))
            using (var images = File.OpenRead("./data/train-images-idx3-ubyte"))
            {
                var data = ReadImages(images, labels).ToArray();

                TrainingData = data.Take(50000).ToArray();
                HoldoutData = data.Skip(50000).ToArray();
            }

            using (var labels = File.OpenRead("./data/t10k-labels-idx1-ubyte"))
            using (var images = File.OpenRead("./data/t10k-images-idx3-ubyte"))
            {
                TestData = ReadImages(images, labels).ToArray();
            }
        }

        private void Download(string filename)
        {
            var ext = new FileInfo(filename).Extension.Length;
            if (!File.Exists($"./data/{filename}"))
            {
                using (var client = new HttpClient())
                {
                    var inStream = 
                        client.GetStreamAsync($"http://yann.lecun.com/exdb/mnist/{filename}").Result;

                    using (var f = File.OpenWrite($"./data/{filename.Remove(filename.Length-ext)}"))
                    using (var gz = new GZipStream(inStream, CompressionMode.Decompress))
                    {
                        gz.CopyTo(f);
                    }
                }
            }
        }

        private int ReadBigEndian(BinaryReader reader)
        {
            var bytes = BitConverter.GetBytes(reader.ReadInt32());
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        private IEnumerable<(float[] image, float[] value)> ReadImages(Stream images, Stream labels)
        {
            using (var imageReader = new BinaryReader(images))
            using (var labelReader = new BinaryReader(labels))
            {
                
                if (ReadBigEndian(labelReader) != 0x801)
                    throw new InvalidDataException();
                
                if (ReadBigEndian(imageReader) != 0x803)
                    throw new InvalidDataException();

                var labelCount = ReadBigEndian(labelReader);
                var imageCount = ReadBigEndian(imageReader);
                
                if (labelCount != imageCount)
                    throw new InvalidDataException("Image count does not match label count.");

                var imageHeight = ReadBigEndian(imageReader);
                var imageWidth = ReadBigEndian(imageReader);

                for (var i = 0; i < imageCount; i++)
                {
                    var image = new float[imageWidth * imageHeight];
                    var value = new float[10];

                    for (var j = 0; j < imageWidth * imageHeight; j++)
                    {
                        image[j] = imageReader.ReadByte() / 255f;
                    }

                    value[labelReader.ReadByte()] = 1f;

                    yield return (image, value);
                }
            }
        }
    }
}