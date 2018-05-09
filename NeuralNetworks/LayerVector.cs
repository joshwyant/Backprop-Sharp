using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NeuralNetworks
{
    public class LayerVector : IEnumerable<float>
    {
        public Vector<float>[] Internal { get; }
        public int Length { get; }
        
        public LayerVector(int n)
        {
            Internal = new Vector<float>[(n + Vector<float>.Count - 1) / Vector<float>.Count];
            Length = n;
        }

        public LayerVector(float[] layer)
            : this(layer.Length)
        {
            for (var i = 0; i < Internal.Length; i++)
            {
                if (Vector<float>.Count <= layer.Length - i)
                {
                    Internal[i] = new Vector<float>(layer, i * Vector<float>.Count);
                }
                else
                {
                    var arr = new float[Vector<float>.Count];
                    Array.Copy(layer, i * Vector<float>.Count, arr, 0, layer.Length - i);
                    Internal[i] = new Vector<float>(arr);
                }
            }
        }

        public float this[int index] => Internal[index / Vector<float>.Count][index % Vector<float>.Count];

        private IEnumerable<float> Enumerate()
        {
            int n = 0;
            foreach (var vector in Internal)
            {
                if (n++ >= Length)
                {
                    yield break;
                }
                for (int i = 0; i < Vector<float>.Count; i++)
                {
                    yield return vector[i];
                }
            }
        }
        
        public IEnumerator<float> GetEnumerator()
        {
            return Enumerate().GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}