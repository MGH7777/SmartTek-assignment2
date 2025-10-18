import streamlit as st
import numpy as np
from math import log10
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- UI: compact page & CSS ----------------
st.set_page_config(page_title="DCT + Quantization — Compact", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 0.5rem; padding-bottom: 0.5rem; max-width: 1600px;}
h1, h2, h3 {margin: 0.25rem 0;}
.caption {font-size: 0.85rem; color: #666;}
</style>
""", unsafe_allow_html=True)

st.title("DCT + Quantization (Lossy Compression) — Compact Dashboard")

# ---------------- DCT core ----------------
N = 8
def dct_matrix(n=N):
    C = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        for i in range(n):
            alpha = np.sqrt(1/n) if k == 0 else np.sqrt(2/n)
            C[k, i] = alpha * np.cos((np.pi*(2*i+1)*k)/(2*n))
    return C
C = dct_matrix(N); CT = C.T

def block_process(img, fn):
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float64)
    for r in range(0, h, N):
        for c in range(0, w, N):
            out[r:r+N, c:c+N] = fn(img[r:r+N, c:c+N])
    return out

def dct2_block(b):  return C @ b @ CT
def idct2_block(b): return CT @ b @ C

def make_test_image(n=512):
    x = np.linspace(0, 1, n); y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    img = 0.6*X + 0.4*Y
    img[int(0.1*n):int(0.3*n), int(0.1*n):int(0.35*n)] += 0.4
    rr, cc = np.ogrid[:n, :n]
    circle = (rr - int(0.7*n))**2 + (cc - int(0.3*n))**2 <= (int(0.12*n))**2
    img[circle] -= 0.5
    ch = int(0.18*n)
    checker = (np.indices((ch, ch)).sum(axis=0) % 2)
    checker = (checker*2-1)*0.25
    img[int(0.55*n):int(0.55*n)+ch, int(0.6*n):int(0.6*n)+ch] += checker
    return (np.clip(img, 0, 1)*255).astype(np.float32)

def psnr(x, y, max_val=255.0):
    mse = np.mean((x.astype(np.float64)-y.astype(np.float64))**2)
    return float('inf') if mse == 0 else 20*log10(max_val) - 10*log10(mse)

def nonzero_fraction(M): return np.count_nonzero(M)/M.size

# ---------------- Quantization matrices (baseline) ----------------
def radial_distance(size=N):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.sqrt(yy**2 + xx**2)
R = radial_distance(N); Rn = R/R.max()

# (a) normal: penalize high-freq strongly
Q_high0 = 1 + 50*(Rn**1.5)
# (b) zero low-freq: large near DC
Q_low0  = 1 + 50*((1 - Rn)**2)
# (c) band-pass: keep middle band
mid = np.exp(-((Rn-0.55)**2)/(2*0.12**2))
Q_band0 = 1 + 50*(1 - mid)

# ---------------- Inputs ----------------
uploaded = st.file_uploader("Upload image (optional). If empty, a synthetic image is used.",
                            type=["png","jpg","jpeg"])
img = np.array(Image.open(uploaded).convert("L"), dtype=np.float32) if uploaded else make_test_image(512)

left, right = st.columns([2,1])
with left:
    scenario = st.radio("Scenario",
        ["(a) High-freq → 0", "(b) Low-freq → 0", "(c) Band-pass"],
        horizontal=True)
with right:
    scale = st.slider("Quantization strength (scales Q)", 10, 200, 50)

# Select baseline Q (unscaled) and make a scaled version
Q_map = {"(a) High-freq → 0": Q_high0, "(b) Low-freq → 0": Q_low0, "(c) Band-pass": Q_band0}
Q0 = Q_map[scenario]              # baseline (for fixed display range)
Q  = Q0 * (scale/50.0)            # scaled (used for quantization)

# ---------------- Pipeline for chosen scenario ----------------
centered = img - 128.0
D  = block_process(centered, dct2_block)
Dq = block_process(D, lambda b: np.round(b/Q)*Q)
rec = block_process(Dq, idct2_block) + 128.0
rec = np.clip(rec, 0, 255).astype(np.float32)

# Metrics
ps = psnr(img, rec)
nz = nonzero_fraction(Dq)
cr = 1.0/max(nz, 1e-9)

# ---------------- Q visualization----------------
figQ = plt.figure(figsize=(3,3))
vmax = Q0.max() * 2.0     
plt.imshow(Q, cmap="gray", vmin=0, vmax=vmax)
plt.title("Quantization Matrix (visualization)")
plt.axis("off")

# ---------------- Row 1: Original | Reconstruction | Q ----------------
c1, c2, c3 = st.columns(3, gap="small")
with c1:
    st.image(img.astype(np.uint8), caption="Original", use_container_width=True)
with c2:
    st.image(rec.astype(np.uint8), caption=f"Reconstruction — PSNR={ps:.2f} dB", use_container_width=True)
with c3:
    st.pyplot(figQ, clear_figure=True)

# ---------------- Optional: histograms on one row ----------------
with st.expander("Show DCT coefficient histograms (react to scenario & strength)"):
    def nz_hist_fig(Dq, title):
        fig = plt.figure(figsize=(3,2))
        nzv = np.abs(Dq[Dq != 0]).flatten()
        if nzv.size > 0:
            plt.hist(nzv, bins=100)
            plt.yscale("log")
        plt.title(title, fontsize=10)
        plt.tight_layout()
        return fig
    h1, h2, h3 = st.columns(3, gap="small")
    h1.pyplot(nz_hist_fig(Dq, "Quantized non-zeros (current scenario)"))

# ---------------- Tiny metrics table ----------------
df = pd.DataFrame({
    "Scenario": [scenario],
    "PSNR (dB)": [round(ps, 2)],
    "Non-zero frac": [round(nz, 3)],
    "~CR (×)": [round(cr, 1)],
})
st.dataframe(df, hide_index=True, use_container_width=True)
st.caption("Tip: use a wider browser window or reduce zoom slightly to keep everything on one screen.")
