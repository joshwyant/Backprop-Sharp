using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks
{
    public class Network
    {
        private Random r;
        
        public Func<float, float> Activation { get; set; }
        public Func<float, float> ActivationPrime { get; set; }
        
        public int Layers { get; }

        public int[] Sizes { get; }

        public LayerVector[] Biases { get; private set; }

        public LayerMatrix[] Weights { get; private set; }
        
        public Network(int[] sizes, Random r = null)
        {
            this.r = r ?? new Random();
            Layers = sizes.Length;
            Sizes = sizes;
            Biases = sizes.Skip(1).Select(y => r.CreateNormal(y)).ToArray();
            Weights = sizes.Take(Layers - 1).Zip(sizes.Skip(1), (x, y) => (layerFromSize: x, layerToSize: y))
                .Select(pair => r.CreateNormal(pair.layerFromSize, pair.layerToSize)).ToArray();
            Activation = VectorUtilities.Sigmoid;
            ActivationPrime = VectorUtilities.SigmoidPrime;
        }

        public void StochasticGradientDescent((float[] x, float[] y)[] trainingData, int epochs, int miniBatchSize, 
            float eta, (float[] x, float[] y)[] testData)
        {
            var batches = (trainingData.Length + miniBatchSize - 1) / miniBatchSize;
            for (var j = 0; j < epochs; j++)
            {
                r.Shuffle(trainingData);
                foreach (var miniBatch in Enumerable.Range(0, batches).Select(batch => batch * miniBatchSize)
                    .Select(k => trainingData.Skip(k).Take(miniBatchSize).ToArray()))
                {
                    UpdateMiniBatch(miniBatch, eta);
                }
                Console.WriteLine(testData != null
                    ? $"Epoch {j}: {Evaluate(testData)} / {trainingData.Length}"
                    : $"Epoch {j} complete");
            }
        }

        private int Evaluate((float[] x, float[] y)[] testData)
        {
            return testData.Select(pair => (x: FeedForward(new LayerVector(pair.x)).ArgMax().ToArray(), y: pair.y))
                .Sum(pair => pair.x.SequenceEqual(pair.y) ? 1 : 0);
        }

        private void UpdateMiniBatch((float[] x, float[] y)[] miniBatch, float eta)
        {
            var nablaB = Biases.Select(b => new LayerVector(b.Length)).ToArray();
            var nablaW = Weights.Select(w => new LayerMatrix(w.Width, w.Height)).ToArray();
            foreach (var pair in miniBatch)
            {
                var (deltaNablaB, deltaNablaW) = Backprop(pair.x, pair.y);
                nablaB = nablaB.Zip(deltaNablaB, (nb, dnb) => (nb: nb, dnb: dnb))
                    .Select(nbdnb => nbdnb.nb.Add(nbdnb.dnb)).ToArray();
                nablaW = nablaW.Zip(deltaNablaW, (nw, dnw) => (nw: nw, dnw: dnw))
                    .Select(nwdnw => nwdnw.nw.Add(nwdnw.dnw)).ToArray();
            }
            Weights = Weights.Zip(nablaW, (x, y) => (w: x, nw: y))
                .Select(pair => pair.w.Subtract(pair.nw.Scale(eta / miniBatch.Length))).ToArray();
            Biases = Biases.Zip(nablaB, (x, y) => (b: x, nb: y))
                .Select(pair => pair.b.Subtract(pair.nb.Multiply(eta / miniBatch.Length))).ToArray();
        }

        private (LayerVector[], LayerMatrix[]) Backprop(float[] x, float[] y)
        {
            var nablaB = Biases.Select(b => new LayerVector(b.Length)).ToArray();
            var nablaW = Weights.Select(w => new LayerMatrix(w.Width, w.Height)).ToArray();
            var activation = new LayerVector(x);
            var activations = new List<LayerVector>(new[] { activation });
            var zs = new List<LayerVector>();
            foreach (var bw in Biases.Zip(Weights, (b, w) => (b: b, w: w)))
            {
                var z = bw.b.Add(activation.Transform(bw.w));
                zs.Add(z);
                activation = z.Apply(Activation);
                activations.Add(activation);
            }
            var delta = CostDerivative(activations[activations.Count - 1], y)
                .Multiply(zs[zs.Count - 1].Apply(ActivationPrime));

            nablaB[nablaB.Length - 1] = delta;
            // Below: multiply delta vector as a matrix with the transpose of the activation of the L-1 layer.
            nablaW[nablaW.Length - 1] = delta.Multiply(activations[activations.Count - 2].Transpose());

            for (var l = 2; l < Layers; l++)
            {
                var z = zs[zs.Count - 1];
                var sp = z.Apply(ActivationPrime);
                // Below: Multiply transpose of the weight matrix with the delta vector as a matrix, 
                // which returns a vector (matrix Mx1), and store the hadamard (elementwise) product
                // of that vector and the sigmoid prime vector back into the delta variable.
                delta = Weights[Weights.Length - l + 1].Transpose().Multiply(delta).Multiply(sp);
                nablaB[nablaB.Length - l] = delta;
                nablaW[nablaW.Length - l] = delta.Multiply(activations[activations.Count - l - 1].Transpose());
            }

            return (nablaB, nablaW);
        }

        private LayerVector CostDerivative(LayerVector outputActivations, float[] y)
        {
            return outputActivations.Subtract(new LayerVector(y));
        }

        public LayerVector FeedForward(LayerVector a)
        {
            foreach (var (b, w) in Biases.Zip(Weights, (b, w) => (b, w)))
            {
                a = a.Transform(w).Add(b).Apply(Activation);
            }
            return a;
        }
    }
}