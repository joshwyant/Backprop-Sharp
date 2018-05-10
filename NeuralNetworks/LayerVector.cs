using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;

namespace NeuralNetworks
{
    public class LayerVector : IEnumerable<float>
    {
        public float[] Internal { get; }
        public int Length { get; }
        
        public LayerVector(int n)
        {
            Internal = new float[n];
            Length = n;
        }

        public LayerVector(float[] layer)
        {
            Internal = layer;
            Length = layer.Length;
        }

        public float this[int index] => Internal[index];
        
        public IEnumerator<float> GetEnumerator()
        {
            return (Internal as IEnumerable<float>).GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public void GetVector(int index, out Vector<float> v)
        {
            if (Length - index < Vector<float>.Count)
            {
                var dest = new float[Vector<float>.Count];
                for (int i = index, j = 0; i < Length; i++, j++)
                {
                    dest[j] = Internal[i];
                }
                v = new Vector<float>(dest);
            }
            else
            {
                v = new Vector<float>(Internal, index);
            }
        }

        public void SetVector(int index, ref Vector<float> v)
        {
            for (int i = index, j = 0; i < index + Vector<float>.Count && i < Length; i++, j++)
            {
                Internal[i] = v[j];
            }
        }
    }
}