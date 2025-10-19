from typing import List, Tuple
import numpy as np
from PIL import Image
import streamlit as st

# ---------- Utils ----------
def pil_to_array(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    y = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return y

def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(u8, mode="L")

def pad_edge(img: np.ndarray, pad: int, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (pad, pad)
    return np.pad(img, pad_width, mode="edge")

# ---------- Gaussian & Convolution ----------
def gaussian_kernel_1d(sigma: float, size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= np.sum(k)
    return k.astype(np.float32)

def convolve_separable(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half = len(kernel) // 2
    # horizontal
    padded = pad_edge(img, half, axis=1)
    out_h = np.zeros_like(img)
    for i in range(-half, half + 1):
        out_h += kernel[i + half] * padded[:, i + half : i + half + img.shape[1]]
    # vertical
    padded_v = pad_edge(out_h, half, axis=0)
    out_v = np.zeros_like(img)
    for j in range(-half, half + 1):
        out_v += kernel[j + half] * padded_v[j + half : j + half + img.shape[0], :]
    return out_v

def gaussian_blur(img: np.ndarray, sigma: float, size: int) -> np.ndarray:
    return convolve_separable(img, gaussian_kernel_1d(sigma, size))

# ---------- Pyramid ops ----------
def downsample_by_2(img: np.ndarray) -> np.ndarray:
    return img[::2, ::2]

def upsample_by_2(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    h, w = img.shape
    H, W = target_shape
    up = np.zeros((H, W), dtype=np.float32)
    up[:, :] = img[
        np.minimum(np.arange(H) // 2, h - 1)[:, None],
        np.minimum(np.arange(W) // 2, w - 1)[None, :],
    ]
    # gentle blur to mimic Gaussian expand
    return gaussian_blur(up, sigma=1.0, size=5)

def build_gaussian_pyramid(base: np.ndarray, levels: int, sigma: float, ksize: int) -> List[np.ndarray]:
    G = [base]
    cur = base
    for _ in range(1, levels):
        cur = downsample_by_2(gaussian_blur(cur, sigma, ksize))
        G.append(cur)
        if min(cur.shape) <= 4:
            break
    return G

def build_laplacian_pyramid(G: List[np.ndarray]) -> List[np.ndarray]:
    L = []
    for i in range(len(G) - 1):
        expanded = upsample_by_2(G[i + 1], G[i].shape)[: G[i].shape[0], : G[i].shape[1]]
        L.append(G[i] - expanded)
    L.append(G[-1])  # smallest Gaussian
    return L

def reconstruct_from_laplacian(L: List[np.ndarray]) -> np.ndarray:
    cur = L[-1]
    for i in range(len(L) - 2, -1, -1):
        expanded = upsample_by_2(cur, L[i].shape)[: L[i].shape[0], : L[i].shape[1]]
        cur = expanded + L[i]
    return cur

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Gaussian & Laplacian Pyramid", layout="wide")
st.title("Gaussian & Laplacian Pyramid — Pure Python/NumPy")

with st.sidebar:
    st.header("Controls")
    levels = st.slider("Levels", 2, 8, 5, 1)
    sigma = st.slider("Gaussian σ (pre-blur before downsample)", 0.5, 3.0, 1.2, 0.1)
    ksize = st.slider("Kernel size (odd)", 3, 21, 5, 2)
    normalize_lap = st.checkbox("Normalize Laplacian for display", value=True)

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("Please upload an image to begin.")
    st.stop()

img = Image.open(uploaded)
st.subheader("Input Image")
st.image(img, caption=f"Original ({img.size[0]}×{img.size[1]})", use_container_width=True)

base = pil_to_array(img)
G = build_gaussian_pyramid(base, levels, sigma, ksize)
L = build_laplacian_pyramid(G)
recon = reconstruct_from_laplacian(L)
err = rmse(base, recon)

# --- simple visualization helper ---
def vis_stack(images: List[np.ndarray], normalize=False) -> Image.Image:
    shown = []
    for x in images:
        x = x.copy()
        if normalize:
            mn, mx = float(np.min(x)), float(np.max(x))
            x = (x - mn) / (mx - mn) if (mx - mn) > 1e-8 else np.zeros_like(x)
        else:
            x = np.clip(x, 0.0, 1.0)
        shown.append(array_to_pil(x))
    widths = [im.width for im in shown]
    heights = [im.height for im in shown]
    H = max(heights)
    W = sum(widths) + (len(shown) - 1) * 4
    canvas = Image.new("L", (W, H), color=0)
    xoff = 0
    for im in shown:
        canvas.paste(im, (xoff, 0))
        xoff += im.width + 4
    return canvas

st.subheader("Gaussian Pyramid")
st.image(vis_stack(G, normalize=False), caption="Left = base → Right = smallest", use_container_width=True)

st.subheader("Laplacian Pyramid")
st.image(
    vis_stack(L, normalize=normalize_lap),
    caption="Left = finest residual → Right = coarsest (last Gaussian)",
    use_container_width=True,
)

st.subheader("Reconstruction")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(array_to_pil(base), caption="Original (grayscale)", use_container_width=True)
with c2:
    st.image(array_to_pil(recon), caption="Reconstructed", use_container_width=True)
with c3:
    diff = np.clip((recon - base) * 0.5 + 0.5, 0.0, 1.0)
    st.image(array_to_pil(diff), caption="Difference (centered)", use_container_width=True)

st.markdown(f"**Reconstruction RMSE:** `{err:.6f}`")

if __name__ == "__main__":
    print("Launch with: streamlit run <this_file>.py")
