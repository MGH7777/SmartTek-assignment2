using System;
using System.Drawing;
using System.Windows.Forms;

namespace CompressionApp
{
    public class MainForm : Form
    {
        // UI containers
        private Panel topPanel = new();
        private FlowLayoutPanel controlsFlow = new(); // left side controls
        private FlowLayoutPanel metricsFlow = new();  // right side metrics

        // Controls
        private Button loadButton = new();
        private ComboBox algoBox = new();
        private TrackBar qualityBar = new();
        private Label qualityLabel = new();
        private CheckBox useHuffmanCheck = new();
        private Button compressButton = new();
        private Label metricsLabel = new();

        // Image area
        private TableLayoutPanel imageLayout = new();
        private PictureBox originalBox = new();
        private PictureBox compressedBox = new();

        private Bitmap? originalImage;

        public MainForm()
        {
            Text = "Lossy Compression (FFT & DCT)";
            MinimumSize = new Size(1200, 750);
            Width = 1400; Height = 800;
            StartPosition = FormStartPosition.CenterScreen;

            BuildUI();
        }

        private void BuildUI()
        {
            // -------- Top panel (header) --------
            topPanel.Dock = DockStyle.Top;
            topPanel.Height = 70;
            topPanel.Padding = new Padding(12, 10, 12, 8);

            // Left flow (controls)
            controlsFlow.Dock = DockStyle.Left;
            controlsFlow.AutoSize = true;
            controlsFlow.AutoSizeMode = AutoSizeMode.GrowAndShrink;
            controlsFlow.WrapContents = false;
            controlsFlow.FlowDirection = FlowDirection.LeftToRight;
            controlsFlow.Padding = new Padding(0);
            controlsFlow.Margin = new Padding(0);

            // Right flow (metrics)
            metricsFlow.Dock = DockStyle.Right;
            metricsFlow.AutoSize = true;
            metricsFlow.AutoSizeMode = AutoSizeMode.GrowAndShrink;
            metricsFlow.WrapContents = false;
            metricsFlow.FlowDirection = FlowDirection.LeftToRight;
            metricsFlow.Padding = new Padding(0);
            metricsFlow.Margin = new Padding(0);

            // Controls: Load
            loadButton.Text = "Load";
            loadButton.AutoSize = true;
            loadButton.Margin = new Padding(0, 0, 8, 0);
            loadButton.Click += LoadImage;

            // Algo dropdown
            algoBox.Items.AddRange(new[] { "DCT", "FFT" });
            algoBox.SelectedIndex = 0;
            algoBox.DropDownStyle = ComboBoxStyle.DropDownList;
            algoBox.Width = 100;
            algoBox.Margin = new Padding(0, 0, 12, 0);

            // Quality slider + label
            qualityBar.Minimum = 5;
            qualityBar.Maximum = 95;
            qualityBar.TickFrequency = 10;
            qualityBar.Value = 75;
            qualityBar.Width = 260;
            qualityBar.Margin = new Padding(0, 4, 6, 0);
            qualityBar.ValueChanged += (_, __) => UpdateQualityLabel();

            qualityLabel.Text = "Quality: 75%";
            qualityLabel.AutoSize = true;
            qualityLabel.Margin = new Padding(0, 8, 16, 0);

            // Huffman toggle
            useHuffmanCheck.Text = "Use Huffman";
            useHuffmanCheck.Checked = true;
            useHuffmanCheck.AutoSize = true;
            useHuffmanCheck.Margin = new Padding(0, 6, 16, 0);

            // Compress button
            compressButton.Text = "Compress";
            compressButton.AutoSize = true;
            compressButton.Margin = new Padding(0, 0, 0, 0);
            compressButton.Click += CompressImage;

            // Metrics label (right)
            metricsLabel.AutoSize = true;
            metricsLabel.Font = new Font("Segoe UI", 10, FontStyle.Bold);
            metricsLabel.ForeColor = Color.DarkBlue;
            metricsLabel.Margin = new Padding(0, 6, 0, 0);

            // Assemble header
            controlsFlow.Controls.AddRange(new Control[]
            {
                loadButton, algoBox, qualityBar, qualityLabel, useHuffmanCheck, compressButton
            });
            metricsFlow.Controls.Add(metricsLabel);

            topPanel.Controls.Add(controlsFlow);
            topPanel.Controls.Add(metricsFlow);

            // -------- Image area --------
            imageLayout.Dock = DockStyle.Fill;
            imageLayout.ColumnCount = 2;
            imageLayout.RowCount = 1;
            imageLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            imageLayout.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 50));
            imageLayout.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            imageLayout.Padding = new Padding(16); // spacing around image area
            imageLayout.BackColor = SystemColors.Control; // contrast from header

            originalBox.Dock = DockStyle.Fill;
            originalBox.SizeMode = PictureBoxSizeMode.Zoom;
            originalBox.Margin = new Padding(16);

            compressedBox.Dock = DockStyle.Fill;
            compressedBox.SizeMode = PictureBoxSizeMode.Zoom;
            compressedBox.Margin = new Padding(16);

            imageLayout.Controls.Add(originalBox, 0, 0);
            imageLayout.Controls.Add(compressedBox, 1, 0);

            // -------- Form layout --------
            Controls.Add(imageLayout);
            Controls.Add(topPanel);

            // Initial metrics text
            metricsLabel.Text = "PSNR: —     SSIM: —     CR: —     Quality: —";
        }

        private void UpdateQualityLabel()
        {
            qualityLabel.Text = $"Quality: {qualityBar.Value}%";
        }

        private void LoadImage(object? sender, EventArgs e)
        {
            using OpenFileDialog ofd = new()
            {
                Filter = "Image files|*.png;*.jpg;*.jpeg;*.bmp;*.dicom;*.dcm"
            };
            if (ofd.ShowDialog() == DialogResult.OK)
            {
                originalImage = ImageUtils.LoadImage(ofd.FileName);
                originalBox.Image = originalImage;
                metricsLabel.Text = "PSNR: —     SSIM: —     CR: —     Quality: —";
            }
        }

        private void CompressImage(object? sender, EventArgs e)
        {
            if (originalImage == null)
            {
                MessageBox.Show("Please load an image first.");
                return;
            }

            string? method = algoBox.SelectedItem?.ToString();
            if (method == null)
            {
                MessageBox.Show("Please select an algorithm first.");
                return;
            }

            int quality = qualityBar.Value;
            bool useHuffman = useHuffmanCheck.Checked;

            var (compressedImg, psnr, compressionRatio) =
                CompressionPipeline.Run(originalImage, method, quality / 100.0, useHuffman);

            compressedBox.Image = compressedImg;

            double ssim = ImageUtils.CalculateSSIM(originalImage, compressedImg);

            string crText = double.IsNaN(compressionRatio)
                ? "CR: — (Huffman off)"
                : $"CR: {compressionRatio:F2}x";

            metricsLabel.Text =
                $"PSNR: {psnr:F2} dB     SSIM: {ssim:F4}     {crText}     Quality: {quality}%";
        }
    }
}
