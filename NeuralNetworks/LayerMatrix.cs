using System.Linq;
using System.Numerics;

namespace NeuralNetworks
{
    public class LayerMatrix
    {
        public readonly LayerVector[] Rows;
        public readonly int Width;
        public readonly int Height;
        
        public LayerMatrix(int w, int h)
        {
            Width = w;
            Height = h;
            Rows = new LayerVector[h];
            for (var j = 0; j < h; j++)
            {
                Rows[j] = new LayerVector(w);
            }
        }

        public LayerMatrix(LayerVector[] rows)
        {
            Width = rows[0].Length;
            Height = rows.Length;
            Rows = rows;
        }
    }
}