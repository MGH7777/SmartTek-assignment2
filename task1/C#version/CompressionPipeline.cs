using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace CompressionApp
{
    public static class CompressionPipeline
    {
        public static (Bitmap compressed, double psnr, double compressionRatio)
            Run(Bitmap input, string method, double quality, bool useHuffman)
        {
            double[,] gray = ImageUtils.ToGrayscaleArray(input);

            double[,] compressedArray;
            double compressionRatio;

            if (method == "FFT")
                compressedArray = CompressFFT(gray, quality, useHuffman, out compressionRatio);
            else if (method == "DCT")
                compressedArray = CompressDCT(gray, quality, useHuffman, out compressionRatio);
            else
                throw new ArgumentException("Unknown compression method.");

            Bitmap compressedImage = ImageUtils.FromGrayscaleArray(compressedArray);
            double psnr = ImageUtils.CalculatePSNR(input, compressedImage);
            return (compressedImage, psnr, compressionRatio);
        }

        // FFT path 
        private static double[,] CompressFFT(double[,] input, double quality, bool useHuffman, out double compressionRatio)
        {
            int h = input.GetLength(0);
            int w = input.GetLength(1);

            Complex[,] F = new Complex[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    F[y, x] = new Complex(input[y, x], 0);

            FFT2D(F, true);

            double threshold = (1.0 - quality) * 50.0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (F[y, x].Magnitude < threshold) F[y, x] = Complex.Zero;

            if (useHuffman)
            {
                double[,] mag = new double[h, w];
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        mag[y, x] = F[y, x].Magnitude;
                compressionRatio = ComputeHuffmanCompressionRatio(mag);
            }
            else
            {
                compressionRatio = double.NaN;
            }

            FFT2D(F, false);

            double[,] output = new double[h, w];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    output[y, x] = Math.Clamp(F[y, x].Real, 0, 1);

            return output;
        }

        // DCT path
        private static double[,] CompressDCT(double[,] input, double quality01, bool useHuffman, out double compressionRatio)
        {
            const int N = 8;
            int h = input.GetLength(0);
            int w = input.GetLength(1);

            double[,] output = new double[h, w];
            var allQuantized = new List<int>();  

            int[,] Qbase = {
                {16,11,10,16,24,40,51,61},
                {12,12,14,19,26,58,60,55},
                {14,13,16,24,40,57,69,56},
                {14,17,22,29,51,87,80,62},
                {18,22,37,56,68,109,103,77},
                {24,35,55,64,81,104,113,92},
                {49,64,78,87,103,121,120,101},
                {72,92,95,98,112,100,103,99}
            };

            int QF = (int)Math.Round(Math.Clamp(quality01, 0.0, 1.0) * 99.0) + 1;
            int S = (QF < 50) ? (5000 / QF) : (200 - 2 * QF);
            double[,] Q = new double[N, N];
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    Q[i, j] = Math.Max(1, Math.Min(255, (Qbase[i, j] * S + 50) / 100));

            for (int by = 0; by < h; by += N)
            {
                for (int bx = 0; bx < w; bx += N)
                {
                    var b = GetBlockWithPadding(input, by, bx, N);

                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++)
                            b[i, j] = b[i, j] * 255.0 - 128.0;

                    var dct = ApplyDCT(b);

                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++)
                            dct[i, j] = Math.Round(dct[i, j] / Q[i, j]);

                    if (useHuffman)
                        allQuantized.AddRange(dct.Cast<double>().Select(v => (int)v));

                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++)
                            dct[i, j] = dct[i, j] * Q[i, j];

                    var idct = ApplyIDCT(dct);

                    for (int i = 0; i < N; i++)
                        for (int j = 0; j < N; j++)
                            idct[i, j] = Math.Clamp((idct[i, j] + 128.0) / 255.0, 0.0, 1.0);

                    PasteBlockClipped(output, idct, by, bx);
                }
            }

                if (useHuffman)
                {
                    int[] symbols = allQuantized.ToArray();

                    if (symbols.Length > 0)
                    {
                        var (codeTable, bitstream) = Huffman.Encode(symbols);

                        // Assume coefficients would otherwise be stored as 16-bit ints
                        double originalBits   = 16.0 * symbols.Length;
                        double compressedBits = bitstream.Length; // one char per bit

                        compressionRatio = originalBits / Math.Max(1.0, compressedBits);
                    }
                    else
                    {
                        compressionRatio = double.NaN;
                    }
                }
                else
                {
                    compressionRatio = double.NaN;
                }

                return output;
            }


        private static void FFT2D(Complex[,] data, bool forward)
        {
            int h = data.GetLength(0);
            int w = data.GetLength(1);

            for (int y = 0; y < h; y++)
            {
                Complex[] row = new Complex[w];
                for (int x = 0; x < w; x++)
                    row[x] = data[y, x];
                if (forward) Fourier.Forward(row, FourierOptions.Matlab);
                else Fourier.Inverse(row, FourierOptions.Matlab);
                for (int x = 0; x < w; x++)
                    data[y, x] = row[x];
            }

            for (int x = 0; x < w; x++)
            {
                Complex[] col = new Complex[h];
                for (int y = 0; y < h; y++)
                    col[y] = data[y, x];
                if (forward) Fourier.Forward(col, FourierOptions.Matlab);
                else Fourier.Inverse(col, FourierOptions.Matlab);
                for (int y = 0; y < h; y++)
                    data[y, x] = col[y];
            }
        }

        private static double[,] ApplyDCT(double[,] block)
        {
            int N = block.GetLength(0);
            double[,] result = new double[N, N];
            for (int u = 0; u < N; u++)
            for (int v = 0; v < N; v++)
            {
                double sum = 0;
                for (int x = 0; x < N; x++)
                for (int y = 0; y < N; y++)
                    sum += block[x, y] *
                           Math.Cos((2 * x + 1) * u * Math.PI / (2 * N)) *
                           Math.Cos((2 * y + 1) * v * Math.PI / (2 * N));
                double cu = (u == 0) ? 1 / Math.Sqrt(2) : 1.0;
                double cv = (v == 0) ? 1 / Math.Sqrt(2) : 1.0;
                result[u, v] = 0.25 * cu * cv * sum;
            }
            return result;
        }

        private static double[,] ApplyIDCT(double[,] block)
        {
            int N = block.GetLength(0);
            double[,] result = new double[N, N];
            for (int x = 0; x < N; x++)
            for (int y = 0; y < N; y++)
            {
                double sum = 0;
                for (int u = 0; u < N; u++)
                for (int v = 0; v < N; v++)
                {
                    double cu = (u == 0) ? 1 / Math.Sqrt(2) : 1.0;
                    double cv = (v == 0) ? 1 / Math.Sqrt(2) : 1.0;
                    sum += cu * cv * block[u, v] *
                           Math.Cos((2 * x + 1) * u * Math.PI / (2 * N)) *
                           Math.Cos((2 * y + 1) * v * Math.PI / (2 * N));
                }
                result[x, y] = 0.25 * sum;
            }
            return result;
        }

        private static double ComputeHuffmanCompressionRatio(double[,] data)
        {
            int[] values = data.Cast<double>().Select(v => (int)Math.Round(v)).ToArray();
            if (values.Length == 0) return 1.0;

            var freq = values.GroupBy(v => v).ToDictionary(g => g.Key, g => g.Count());
            double total = freq.Values.Sum();
            double entropy = 0.0;

            foreach (var f in freq.Values)
            {
                double p = f / total;
                entropy += -p * Math.Log(p, 2);
            }

            double avgBits = Math.Max(entropy, 0.1);
            return 8.0 / avgBits;
        }

        private static double[,] GetBlockWithPadding(double[,] input, int startY, int startX, int blockSize)
        {
            int h = input.GetLength(0);
            int w = input.GetLength(1);
            double[,] block = new double[blockSize, blockSize];

            for (int i = 0; i < blockSize; i++)
            {
                int sy = Math.Min(startY + i, h - 1);
                for (int j = 0; j < blockSize; j++)
                {
                    int sx = Math.Min(startX + j, w - 1);
                    block[i, j] = input[sy, sx];
                }
            }
            return block;
        }

        private static void PasteBlockClipped(double[,] dest, double[,] block, int startY, int startX)
        {
            int h = dest.GetLength(0);
            int w = dest.GetLength(1);
            int bs = block.GetLength(0);

            for (int i = 0; i < bs; i++)
            {
                int dy = startY + i; if (dy >= h) break;
                for (int j = 0; j < bs; j++)
                {
                    int dx = startX + j; if (dx >= w) break;
                    dest[dy, dx] = block[i, j];
                }
            }
        }
    }

    public static class ArrayExtensions
    {
        public static double[,] ToArray2D(this IEnumerable<double> source, int height, int width)
        {
            double[,] result = new double[height, width];
            int i = 0;
            foreach (var v in source)
            {
                if (i >= width) break;
                result[0, i++] = v;
            }
            return result;
        }
    }
}
