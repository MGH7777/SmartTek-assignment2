# Gaussian & Laplacian Pyramid (Streamlit)

Pure NumPy + Streamlit app (no OpenCV). Upload an image, then view Gaussian and Laplacian pyramids. Uses custom functions for kernel, convolution, blur, (down/up)sample, and pyramids. (Optional: shows reconstruction + RMSE.)

## Setup
All assignment2 have the same environment

conda env create -f environment.yml
conda activate smarttek2

## Run
streamlit run app3.py

## Use
Upload image (png/jpg/jpeg/bmp/tif).

Adjust Levels, σ, Kernel size.

See Gaussian (left→right smaller) and Laplacian (fine→coarse).

If enabled, Reconstruction should match original; RMSE ≈ 0.