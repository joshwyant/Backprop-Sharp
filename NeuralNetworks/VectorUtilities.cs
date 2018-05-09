using System;
using System.Linq;
using System.Numerics;

namespace NeuralNetworks
{
    public static class VectorUtilities
    {
        public static LayerVector Transform(this LayerVector value, LayerMatrix matrix)
        {
            if (value.Length != matrix.Width)
                throw new ArgumentException();
            
            var array = new float[matrix.Height];
            for (var j = 0; j < matrix.Height; j++)
            {
                array[j] = value.Dot(matrix.Rows[j]);
            }
            return new LayerVector(array);
        }

        public static LayerMatrix Transpose(this LayerMatrix m)
        {   
            var floats = Enumerable.Range(0, m.Height)
                .Select(i => new float[m.Width]).ToArray();

            for (var i = 0; i < m.Height; i++)
            {
                for (var j = 0; j < m.Width; j++)
                {
                    floats[i][j] = m.Rows[j][i];
                }
            }
            
            return new LayerMatrix(floats.Select(f => new LayerVector(f)).ToArray());
        }

        public static LayerMatrix Multiply(this LayerMatrix m, LayerMatrix other)
        {
            if (m.Width != other.Height) // Inner dimensions must match.
                throw new ArgumentException();
            
            var floats = Enumerable.Range(0, m.Height)
                .Select(i => new float[m.Width]).ToArray();
            
            // Transpose the other matrix to make multiplication faster, since we implemented rows as vectors.
            var otherTransposed = other.Transpose();

            for (var i = 0; i < m.Height; i++)
            {
                for (var j = 0; j < m.Width; j++)
                {
                    floats[i][j] = m.Rows[i].Dot(otherTransposed.Rows[j]);
                }
            }
            
            return new LayerMatrix(floats.Select(f => new LayerVector(f)).ToArray());
        }

        public static LayerMatrix AsMatrix(this LayerVector v)
        {
            return new LayerMatrix(v.Select(f => new LayerVector(new[] { f })).ToArray());
        }

        public static LayerVector AsVector(this LayerMatrix m)
        {
            if (m.Width != 1)
                throw new ArgumentException();
            
            return new LayerVector(m.Rows.Select(r => r[0]).ToArray());
        }

        /// <summary>
        /// Turns the vector into an Nx1 matrix, and multiplies it by the other matrix.
        /// This is NOT the same thing as transforming a vector by a matrix!
        /// The other matrix must be a 1xM matrix.
        /// Ther resulting matrix is size NxM.
        /// </summary>
        public static LayerMatrix Multiply(this LayerVector v, LayerMatrix other)
        {
            return v.AsMatrix().Multiply(other);
        }

        /// <summary>
        /// Multiplies the matrix by an Mx1 matrix representing the other vector, returning a vector.
        /// This is NOT the same thing as transforming a vector by a matrix!
        /// The original matrix must be an NxM matrix.
        /// Ther resulting matrix is size Nx1. The result is returned as a vector of size N.
        /// </summary>
        public static LayerVector Multiply(this LayerMatrix m, LayerVector other)
        {
            return m.Multiply(other.AsMatrix()).AsVector();
        }

        public static LayerMatrix Transpose(this LayerVector v)
        {
            return new LayerMatrix(new[] { v });
        }

        public static LayerMatrix Add(this LayerMatrix m, LayerMatrix n)
        {
            return m.Apply(n, Vector.Add);
        }

        public static LayerMatrix Subtract(this LayerMatrix m, LayerMatrix n)
        {
            return m.Apply(n, Vector.Subtract);
        }

        public static LayerMatrix Scale(this LayerMatrix m, float x)
        {
            return m.Apply(x, Vector.Multiply);
        }

        public static LayerMatrix Divide(this LayerMatrix m, float x)
        {
            return m.Apply(1f/x, Vector.Multiply);
        }

        public static LayerMatrix Apply(this LayerMatrix m, LayerMatrix n, 
            Func<Vector<float>, Vector<float>, Vector<float>> f)
        {
            if (m.Width != n.Width || m.Height != n.Height)
                throw new ArgumentException();
            
            var added = m.Rows.Zip(n.Rows, (x, y) => (row1: x, row2: y)).Select(rows => rows.row1.Apply(rows.row2, f))
                .ToArray();
            
            return new LayerMatrix(added);
        }

        public static LayerMatrix Apply(this LayerMatrix m, float x, 
            Func<Vector<float>, float, Vector<float>> f)
        {
            var added = m.Rows.Select(row => row.Apply(x, f)).ToArray();
            
            return new LayerMatrix(added);
        }

        public static LayerVector Apply(this LayerVector value, Func<Vector<float>, Vector<float>> f)
        {
            var vector = new LayerVector(value.Length);
            for (var i = 0; i < vector.Internal.Length; i++)
            {
                vector.Internal[i] = f(value.Internal[i]);
            }
            return vector;
        }

        public static LayerVector Apply(this LayerVector value, LayerVector other,
            Func<Vector<float>, Vector<float>, Vector<float>> f)
        {
            if (value.Length != other.Length)
                throw new ArgumentException();
            
            var vector = new LayerVector(value.Length);
            for (var i = 0; i < vector.Internal.Length; i++)
            {
                vector.Internal[i] = f(value.Internal[i], other.Internal[i]);
            }
            return vector;
        }

        public static LayerVector Apply(this LayerVector value, float x, Func<Vector<float>, float, Vector<float>> f)
        {
            var vector = new LayerVector(value.Length);
            for (var i = 0; i < vector.Internal.Length; i++)
            {
                vector.Internal[i] = f(value.Internal[i], x);
            }
            return vector;
        }

        public static LayerVector Multiply(this LayerVector value, float x)
        {
            return value.Apply(x, Vector.Multiply);
        }

        public static LayerVector Divide(this LayerVector value, float x)
        {
            return value.Apply(1f/x, Vector.Multiply);
        }

        public static float Dot(this LayerVector value, LayerVector other)
        {
            if (value.Length != other.Length)
                throw new ArgumentException();
            
            return value.Internal.Select((t, i) => Vector.Dot(t, other.Internal[i])).Sum();
        }

        public static LayerVector Apply(this LayerVector value, Func<float, float> f)
        {
            return value.Apply(v => v.Apply(f));
        }

        public static LayerVector Add(this LayerVector value, LayerVector other)
        {
            return value.Apply(other, Vector.Add);
        }

        public static LayerVector Subtract(this LayerVector value, LayerVector other)
        {
            return value.Apply(other, Vector.Subtract);
        }

        /// <summary>
        /// Computes the Hadamard product of two vectors (elementwise multiplication)
        /// </summary>
        public static LayerVector Multiply(this LayerVector value, LayerVector other)
        {
            return value.Apply(other, Vector.Multiply);
        }

        public static LayerVector Divide(this LayerVector value, LayerVector other)
        {
            return value.Apply(other, Vector.Divide);
        }

        public static LayerVector Apply(this LayerVector value, Func<double, double> d)
        {
            return value.Apply(x => (float) d(x));
        }
        
        public static Vector<T> Apply<T>(this Vector<T> value, Func<T, T> f)
            where T : struct
        {
            var array = new T[Vector<T>.Count];
            for (var j = 0; j < array.Length; j++)
            {
                array[j] = f(value[j]);
            }
            return new Vector<T>(array);
        }

        public static LayerVector CreateNormal(this Random r, int n, float mean = 0, float stdDev = 1)
        {
            var a = new float[n];
            for (var i = 0; i < n; i++)
            {
                a[i] = r.NextNormal(mean, stdDev);
            }
            return new LayerVector(a);
        }

        public static LayerMatrix CreateNormal(this Random r, int w, int h, float mean = 0, float stdDev = 1)
        {
            var a = new LayerVector[h];
            for (var i = 0; i < h; i++)
            {
                a[i] = r.CreateNormal(w, mean, stdDev);
            }
            return new LayerMatrix(a);
        }

        public static float NextFloat(this Random r) => (float)r.NextDouble();

        public static float NextNormal(this Random r, float mean = 0, float stdDev = 1)
        {
            var u1 = 1.0-r.NextDouble(); //uniform(0,1] random doubles
            var u2 = 1.0-r.NextDouble();
            var randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                                   Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)(mean + stdDev * randStdNormal); //random normal(mean,stdDev^2)
        }

        public static void Shuffle<T>(this Random r, T[] a)
        {
            int i = 0;
            foreach (var elem in a.OrderBy(item => r.Next()))
            {
                a[i++] = elem;
            }
        }

        public static float[] ArgMax(this LayerVector value)
        {
            var values = value.ToList();
            var result = new float[value.Length];
            result[values.IndexOf(values.Max())] = 1f;
            return result;
        }

        public static float Sigmoid(float x) => 1f / (1f + (float) Math.Exp(-x));
        
        public static float SigmoidPrime(float x) => Sigmoid(x) * (1f - Sigmoid(x));
    }
}