# Lossy DCT + Quantization demo with three quantization matrices
# (a) zero high frequencies, (b) zero low frequencies, (c) keep only mid-band
# Measures PSNR and shows reconstructions.

import numpy as np
import matplotlib.pyplot as plt
from math import log10
from pathlib import Path

# ---------- Utility: make a synthetic test image (grayscale, 512x512) ----------
def make_test_image(n=512):
    # base: smooth gradient
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    img = 0.6*X + 0.4*Y

    # add geometric shapes with sharp edges
    img[int(0.1*n):int(0.3*n), int(0.1*n):int(0.35*n)] += 0.4  # bright rectangle
    rr, cc = np.ogrid[:n, :n]
    circle = (rr - int(0.7*n))**2 + (cc - int(0.3*n))**2 <= (int(0.12*n))**2
    img[circle] -= 0.5  # dark circle

    # add a checkerboard patch (high freq region)
    ch = int(0.18*n)
    cs = 8
    checker = np.indices((ch, ch)).sum(axis=0) % 2
    checker = (checker*2-1) * 0.25
    img[int(0.55*n):int(0.55*n)+ch, int(0.6*n):int(0.6*n)+ch] += checker

    # normalize to [0,1]
    img = np.clip(img, 0, 1)
    return (img*255).astype(np.float32)

orig = make_test_image(512)

# ---------- Block DCT helpers (size 8x8) ----------
N = 8
def dct_matrix(n=N):
    C = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            alpha = np.sqrt(1/n) if k == 0 else np.sqrt(2/n)
            C[k, i] = alpha * np.cos((np.pi*(2*i+1)*k)/(2*n))
    return C

C = dct_matrix(N)
CT = C.T

def block_process(img, fn):
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    for r in range(0, h, N):
        for c in range(0, w, N):
            block = img[r:r+N, c:c+N]
            out[r:r+N, c:c+N] = fn(block)
    return out

def dct2_block(block):
    return C @ block @ CT

def idct2_block(block):
    return CT @ block @ C

# ---------- Quantization matrices (our designs) ----------
def radial_distance_mask(size=N):
    yy, xx = np.mgrid[0:size, 0:size]
    # distance from top-left (DC is at 0,0 in DCT block ordering)
    return np.sqrt(yy**2 + xx**2)

R = radial_distance_mask(N)
R_norm = R / R.max()

# (a) "normal" compression: penalize high frequencies strongly
Q_high = 1 + 50*(R_norm**1.5)   # small near DC, large in high-freq corner

# (b) remove low frequencies: huge weights near DC, smaller elsewhere
Q_low = 1 + 50*((1 - R_norm)**2)  # very large near DC, close to 1 near high freq

# (c) band-pass: kill very low and very high, keep middle band
mid = np.exp(-((R_norm-0.55)**2) / (2*0.12**2))  # Gaussian around mid-band
Q_band = 1 + 50*(1 - mid)  # large outside mid band, small in middle

def quantize_coeffs(D, Q):
    # standard JPEG-like: divide by Q, round, multiply back
    return np.round(D / Q) * Q

def compress_decompress(img, Q):
    # center to range around 0 like JPEG (optional but common)
    centered = img - 128.0
    D = block_process(centered, dct2_block)

    # quantize per block
    def q_fn(b):
        return quantize_coeffs(b, Q)
    Dq = block_process(D, q_fn)

    rec_centered = block_process(Dq, idct2_block)
    rec = rec_centered + 128.0
    rec = np.clip(rec, 0, 255).astype(np.float32)
    return rec

# ---------- Run all three scenarios ----------
rec_high = compress_decompress(orig, Q_high)
rec_low  = compress_decompress(orig, Q_low)
rec_band = compress_decompress(orig, Q_band)

# ---------- PSNR ----------
def psnr(x, y, max_val=255.0):
    mse = np.mean((x.astype(np.float64)-y.astype(np.float64))**2)
    if mse == 0:
        return float('inf')
    return 20*log10(max_val) - 10*log10(mse)

ps_high = psnr(orig, rec_high)
ps_low  = psnr(orig, rec_low)
ps_band = psnr(orig, rec_band)

# ---------- Save artifacts ----------
out_dir = Path("/mnt/data/dct_quant_demo")
out_dir.mkdir(parents=True, exist_ok=True)
def save_img(arr, name):
    from imageio.v2 import imwrite
    imwrite(out_dir / name, arr.astype(np.uint8))

save_img(orig, "original.png")
save_img(rec_high, "recon_highfreq_zero.png")
save_img(rec_low,  "recon_lowfreq_zero.png")
save_img(rec_band, "recon_bandpass.png")

# Save quantization matrices as images for visualization (scaled)
def viz_q(Q, name):
    Qn = Q / Q.max()
    plt.figure(figsize=(3,3))
    plt.imshow(Qn, cmap='gray', vmin=0, vmax=1)
    plt.title(name)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}.png", dpi=200)
    plt.close()

viz_q(Q_high, "Q_highfreq")
viz_q(Q_low, "Q_lowfreq")
viz_q(Q_band, "Q_bandpass")

# ---------- Show results ----------
plt.figure(figsize=(5,5))
plt.imshow(orig, cmap='gray', vmin=0, vmax=255)
plt.title("Original (synthetic)")
plt.axis('off')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(rec_high, cmap='gray', vmin=0, vmax=255)
plt.title(f"(a) Normal: zero high-freq — PSNR={ps_high:.2f} dB")
plt.axis('off')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(rec_low, cmap='gray', vmin=0, vmax=255)
plt.title(f"(b) Zero low-freq — PSNR={ps_low:.2f} dB")
plt.axis('off')
plt.show()

plt.figure(figsize=(5,5))
plt.imshow(rec_band, cmap='gray', vmin=0, vmax=255)
plt.title(f"(c) Band-pass (keep mid) — PSNR={ps_band:.2f} dB")
plt.axis('off')
plt.show()

# Show Q visualizations too
plt.figure(figsize=(3,3))
plt.imshow(Q_high / Q_high.max(), cmap='gray', vmin=0, vmax=1)
plt.title("Quant Matrix: (a)")
plt.axis('off')
plt.show()

plt.figure(figsize=(3,3))
plt.imshow(Q_low / Q_low.max(), cmap='gray', vmin=0, vmax=1)
plt.title("Quant Matrix: (b)")
plt.axis('off')
plt.show()

plt.figure(figsize=(3,3))
plt.imshow(Q_band / Q_band.max(), cmap='gray', vmin=0, vmax=1)
plt.title("Quant Matrix: (c)")
plt.axis('off')
plt.show()

# Print PSNRs
print(f"PSNRs (dB):\n (a) High-freq -> 0: {ps_high:.2f}\n (b) Low-freq -> 0: {ps_low:.2f}\n (c) Band-pass: {ps_band:.2f}\n")
print("Saved outputs in:", out_dir)

