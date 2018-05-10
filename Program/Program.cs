using System;
using NeuralNetworks;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var mnist = new MNIST.MNIST();

            var n = new Network(new[] {784, 30, 10});
            n.StochasticGradientDescent(mnist.TrainingData, 30, 10, 3.0f, mnist.TestData);
        }
    }
}