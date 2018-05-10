using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetworks
{
    public static class VectorUtilities
    {
        public static void Shuffle<T>(this Random r, T[] a)
        {
            int i = 0;
            foreach (var elem in a.OrderBy(item => r.Next()))
            {
                a[i++] = elem;
            }
        }

        public static float[] ArgMax(this IEnumerable<float> value)
        {
            var values = value.ToList();
            var result = new float[values.Count];
            result[values.IndexOf(values.Max())] = 1f;
            return result;
        }

        public static float Sigmoid(float x) => 1f / (1f + (float) Math.Exp(-x));
        
        public static float SigmoidPrime(float x) => Sigmoid(x) * (1f - Sigmoid(x));

        public static float Relu(float x) => x > 0 ? x : 0;

        public static float ReluPrime(float x) => x > 0 ? 1 : 0;
    }
}