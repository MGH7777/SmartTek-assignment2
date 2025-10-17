using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Windows.Forms;
using FellowOakDicom;
using FellowOakDicom.Imaging;
using FellowOakDicom.Imaging.Codec;

namespace CompressionApp
{
    public static class ImageUtils
    {
        public static Bitmap LoadImage(string path)
        {
            string ext = Path.GetExtension(path).ToLowerInvariant();
            if (ext == ".dcm" || ext == ".dicom")
                return LoadDicomCTManual(path);   // CT-safe manual decoder

            return new Bitmap(path);
        }

        /// <summary>
        /// Manual CT decoder that handles:
        /// - 16-bit signed pixels (PixelRepresentation==1)
        /// - BitsStored/HighBit alignment
        /// - Rescale Slope/Intercept (to HU)
        /// - Window Center/Width (first value if multi-valued)
        /// - MONOCHROME1 inversion
        /// </summary>
        private static Bitmap LoadDicomCTManual(string path)
        {
            try
            {
                var src = DicomFile.Open(path);
                var transcoder = new DicomTranscoder(src.FileMetaInfo.TransferSyntax, DicomTransferSyntax.ExplicitVRLittleEndian);
                var file = transcoder.Transcode(src);

                var ds = file.Dataset;
                var dp = FellowOakDicom.Imaging.DicomPixelData.Create(ds);

                int width = dp.Width;
                int height = dp.Height;

                ushort bitsAllocated = dp.BitsAllocated;
                ushort bitsStored = dp.BitsStored;
                ushort highBit = dp.HighBit;

                ushort pixelRep = ds.GetSingleValueOrDefault<ushort>(DicomTag.PixelRepresentation, 0);
                bool isSigned = pixelRep == 1;

                string pi = ds.GetSingleValueOrDefault(DicomTag.PhotometricInterpretation, "MONOCHROME2");
                bool invert = pi.Equals("MONOCHROME1", StringComparison.OrdinalIgnoreCase);

                double slope = ds.Contains(DicomTag.RescaleSlope) ? ds.GetSingleValue<double>(DicomTag.RescaleSlope) : 1.0;
                double intercept = ds.Contains(DicomTag.RescaleIntercept) ? ds.GetSingleValue<double>(DicomTag.RescaleIntercept) : 0.0;

                double wc = 40, ww = 400;
                if (ds.Contains(DicomTag.WindowCenter))
                {
                    var a = ds.GetValues<double>(DicomTag.WindowCenter);
                    if (a.Length > 0) wc = a[0];
                }
                if (ds.Contains(DicomTag.WindowWidth))
                {
                    var a = ds.GetValues<double>(DicomTag.WindowWidth);
                    if (a.Length > 0) ww = a[0];
                }
                double wmin = wc - ww / 2.0;
                double wmax = wc + ww / 2.0;

                var frame = dp.GetFrame(0);
                byte[] px = frame.Data;
                int pixelCount = width * height;

                var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
                var rect = new Rectangle(0, 0, width, height);
                var data = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format24bppRgb);

                try
                {
                    int stride = data.Stride;
                    unsafe
                    {
                        byte* dst = (byte*)data.Scan0;

                        if (bitsAllocated == 8)
                        {
                            int available = Math.Min(pixelCount, px.Length);
                            for (int i = 0; i < available; i++)
                            {
                                byte v = px[i];
                                if (invert) v = (byte)(255 - v);
                                int x = i % width, y = i / width;
                                byte* row = dst + y * stride;
                                int idx = x * 3;
                                row[idx + 0] = v; row[idx + 1] = v; row[idx + 2] = v;
                            }
                        }
                        else if (bitsAllocated == 16)
                        {
                            int stored = bitsStored;
                            int msb = highBit;
                            int lsb = msb - stored + 1;
                            if (lsb < 0) { lsb = 0; msb = stored - 1; }
                            int mask = (stored >= 31) ? -1 : ((1 << stored) - 1);

                            int availableSamples = px.Length / 2;
                            int samples = Math.Min(pixelCount, availableSamples);

                            for (int i = 0; i < samples; i++)
                            {
                                int bi = i * 2;
                                ushort raw = (ushort)(px[bi] | (px[bi + 1] << 8));
                                int val = (raw >> lsb) & mask;

                                if (isSigned)
                                {
                                    int signBit = 1 << (stored - 1);
                                    if ((val & signBit) != 0) val -= (1 << stored);
                                }

                                double hu = val * slope + intercept;

                                byte v = (hu <= wmin) ? (byte)0 :
                                         (hu >= wmax) ? (byte)255 :
                                         (byte)((hu - wmin) * 255.0 / (wmax - wmin));

                                if (invert) v = (byte)(255 - v);

                                int x = i % width, y = i / width;
                                byte* row = dst + y * stride;
                                int idx = x * 3;
                                row[idx + 0] = v; row[idx + 1] = v; row[idx + 2] = v;
                            }
                        }
                        else
                        {
                            int available = Math.Min(pixelCount, px.Length);
                            for (int i = 0; i < available; i++)
                            {
                                byte v = px[i];
                                if (invert) v = (byte)(255 - v);
                                int x = i % width, y = i / width;
                                byte* row = dst + y * stride;
                                int idx = x * 3;
                                row[idx + 0] = v; row[idx + 1] = v; row[idx + 2] = v;
                            }
                        }
                    }
                }
                finally
                {
                    bmp.UnlockBits(data);
                }

                return bmp;
            }
            catch (Exception ex)
            {
                MessageBox.Show($"DICOM load failed:\n{ex.Message}", "DICOM Error",
                    MessageBoxButtons.OK, MessageBoxIcon.Error);
                return CreatePlaceholderBitmap(512, 512, "DICOM Load Failed");
            }
        }

        private static Bitmap CreatePlaceholderBitmap(int width, int height, string text)
        {
            var bmp = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            using var g = Graphics.FromImage(bmp);
            g.Clear(Color.DimGray);
            using var f = new Font("Segoe UI", 12, FontStyle.Bold);
            using var b = new SolidBrush(Color.White);
            g.DrawString(text, f, b, new PointF(10, 10));
            return bmp;
        }

        // Convert to grayscale array (0–1)
        public static double[,] ToGrayscaleArray(Bitmap bmp)
        {
            int w = bmp.Width, h = bmp.Height;
            double[,] g = new double[h, w];

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    var c = bmp.GetPixel(x, y);
                    g[y, x] = (0.299 * c.R + 0.587 * c.G + 0.114 * c.B) / 255.0;
                }
            return g;
        }

        public static Bitmap FromGrayscaleArray(double[,] arr)
        {
            int h = arr.GetLength(0), w = arr.GetLength(1);
            var bmp = new Bitmap(w, h, PixelFormat.Format24bppRgb);

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    int v = (int)(Math.Clamp(arr[y, x], 0.0, 1.0) * 255.0);
                    bmp.SetPixel(x, y, Color.FromArgb(v, v, v));
                }
            return bmp;
        }

        // PSNR metric
        public static double CalculatePSNR(Bitmap original, Bitmap compressed)
        {
            if (original.Width != compressed.Width || original.Height != compressed.Height)
                compressed = new Bitmap(compressed, original.Size);

            int w = original.Width, h = original.Height;
            double mse = 0;

            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    var a = original.GetPixel(x, y);
                    var b = compressed.GetPixel(x, y);
                    double d = a.R - b.R;
                    mse += d * d;
                }

            mse /= (w * h);
            if (mse == 0) return double.PositiveInfinity;
            return 10 * Math.Log10((255.0 * 255.0) / mse);
        }

        // SSIM metric
        public static double CalculateSSIM(Bitmap img1, Bitmap img2)
        {
            if (img1.Width != img2.Width || img1.Height != img2.Height)
                img2 = new Bitmap(img2, img1.Size);

            double[,] A = ToGrayscaleArray(img1);
            double[,] B = ToGrayscaleArray(img2);

            int h = img1.Height, w = img1.Width;
            int N = w * h;

            double muA = 0, muB = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    muA += A[y, x];
                    muB += B[y, x];
                }
            muA /= N; muB /= N;

            double varA = 0, varB = 0, covAB = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    double da = A[y, x] - muA;
                    double db = B[y, x] - muB;
                    varA += da * da;
                    varB += db * db;
                    covAB += da * db;
                }

            varA /= N;
            varB /= N;
            covAB /= N;

            double L = 1.0;
            double K1 = 0.01, K2 = 0.03;
            double C1 = (K1 * L) * (K1 * L);
            double C2 = (K2 * L) * (K2 * L);

            double numerator = (2 * muA * muB + C1) * (2 * covAB + C2);
            double denominator = (muA * muA + muB * muB + C1) * (varA + varB + C2);

            return numerator / denominator;
        }
    }
}
