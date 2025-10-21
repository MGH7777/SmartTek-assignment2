# app5.py
# -----------------------------------------------------------------------------
# Wavelets (MRA): "Transport" meta information by exchanging one detail layer.
# Idea:
#   - Take a COVER image.
#   - Compute an L-level 2D DWT (wavelet pyramid).
#   - Choose a level (1..L) and detail band (H/V/D).
#   - Replace or add a PAYLOAD (meta) image into that band.
#   - Inverse DWT -> marked image that visually looks similar to the cover,
#     but carries the payload "transported" in that wavelet layer.
#   - Extract: DWT the marked image, read back that same band.
#
# GUI (Streamlit):
#   - Upload COVER image (or use random).
#   - Upload PAYLOAD image or type TEXT (rendered into a small image).
#   - Select wavelet, levels, target level, band, and embed mode/strength.
#   - Visualize cover, payload, marked image, difference, and extracted payload.
#
# Why this works (short version):
#   Wavelet details separate image content by orientation/scale. By modifying a
#   *single* detail subband (e.g., horizontal details at level 2), we can hide
#   or "transport" structured meta content into that scale/orientation of the
#   image with controlled visual impact.
# -----------------------------------------------------------------------------

import io
from typing import Tuple, List, Any, Dict

import numpy as np
import streamlit as st

# Core deps
try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

from PIL import Image, ImageDraw, ImageFont


# ----------------------- Utils -----------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Return float32 grayscale in [0,1]."""
    if img.ndim == 3 and img.shape[2] == 3:
        if HAVE_CV2:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            g = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)
    elif img.ndim == 2:
        g = img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if g.max() > 1.5:
        g = g / 255.0
    return np.clip(g, 0.0, 1.0).astype(np.float32)

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)

def resize_to(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize 2D array to (H,W)."""
    H, W = target_hw
    if HAVE_CV2:
        return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    # Nearest fallback
    ys = (np.linspace(0, img.shape[0]-1, H)).astype(int)
    xs = (np.linspace(0, img.shape[1]-1, W)).astype(int)
    return img[np.ix_(ys, xs)]

def grid_show(items: List[Tuple[np.ndarray, str]], cols: int = 3):
    """Display a list of (image, caption) side-by-side."""
    if not items:
        return
    chunks = [items[i:i+cols] for i in range(0, len(items), cols)]
    for chunk in chunks:
        row = st.columns(len(chunk), gap="small")
        for (img, cap), col in zip(chunk, row):
            with col:
                col.image(img, caption=cap, clamp=True, use_container_width=True)

def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR between two float [0,1] images."""
    mse = float(np.mean((img1 - img2) ** 2))
    if mse == 0:
        return 99.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))

def heatmap_abs_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return a grayscale heatmap of absolute difference [0,1]."""
    d = np.abs(a - b)
    d = d / (d.max() + 1e-6)
    return d


# ----------------------- Text -> image payload -----------------------

def text_to_image(text: str, size: Tuple[int,int]=(128,128)) -> np.ndarray:
    """Render a short text into a small grayscale image [0,1]."""
    W, H = size[1], size[0]
    img = Image.new("L", (W, H), color=0)
    draw = ImageDraw.Draw(img)

    # Use default font; wrap text to fit
    lines = []
    words = text.split()
    line = ""
    for w in words:
        if len(line + " " + w) < 12:
            line = (line + " " + w).strip()
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)

    y = 10
    for ln in lines[:6]:
        draw.text((8, y), ln, fill=255)
        y += 18

    arr = np.array(img).astype(np.float32) / 255.0
    return arr


# ----------------------- Wavelet helpers -----------------------

def wavedec2_gray(gray: np.ndarray, levels: int, wavelet: str):
    """
    Return pywt.wavedec2 coefficients:
      [cA_L, (cH_L,cV_L,cD_L), (cH_{L-1},...), ..., (cH_1,cV_1,cD_1)]
    """
    if not HAVE_PYWT:
        raise RuntimeError("PyWavelets is required for this app.")
    return pywt.wavedec2(gray, wavelet=wavelet, level=levels)

def waverec2_gray(coeffs, wavelet: str) -> np.ndarray:
    rec = pywt.waverec2(coeffs, wavelet=wavelet)
    rec = np.clip(rec, 0.0, 1.0).astype(np.float32)
    return rec

def get_band_tuple(coeffs, level: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    level: 1..L (1 is finest detail, L is coarsest detail).
    coeffs structure described in wavedec2_gray().
    """
    L = len(coeffs) - 1
    assert 1 <= level <= L
    return coeffs[L - (level - 1)]

def set_band_tuple(coeffs, level: int, new_tuple: Tuple[np.ndarray,np.ndarray,np.ndarray]):
    L = len(coeffs) - 1
    coeffs[L - (level - 1)] = new_tuple

def normalize_like(src: np.ndarray, ref: np.ndarray, mode: str = "match_std") -> np.ndarray:
    """
    Normalize 'src' to be comparable to 'ref' stats.
    mode:
        - 'match_std' : zero-mean then scale to ref std
        - 'scale01'   : scale src to [0,1], then fit range to ref std
    """
    if mode == "match_std":
        s = src - float(np.mean(src))
        rs = float(np.std(ref)) + 1e-6
        s = s / (float(np.std(s)) + 1e-6)
        return s * rs
    else:
        s = src.copy()
        s = (s - s.min()) / (s.max() - s.min() + 1e-6)
        s = s - s.mean()
        return s * (float(np.std(ref)) + 1e-6)


# ----------------------- Embedding / Extraction -----------------------

def embed_payload_in_band(cover_gray: np.ndarray,
                          payload_gray: np.ndarray,
                          levels: int,
                          wavelet: str,
                          target_level: int,    # 1..levels
                          band: str,            # 'H','V','D'
                          mode: str,            # 'replace' or 'add'
                          alpha: float) -> Dict[str, Any]:
    """
    Return dict with: marked_gray, replaced_band (for viz), original_band, payload_used
    """
    coeffs = list(wavedec2_gray(cover_gray, levels, wavelet))
    # Target band shape
    (cH, cV, cD) = get_band_tuple(coeffs, target_level)
    band_map = {'H': 0, 'V': 1, 'D': 2}
    band = band.upper()
    assert band in band_map, "Band must be H, V, or D"
    idx = band_map[band]

    # Resize payload to target band size
    band_shape = (cH.shape[0], cH.shape[1])
    payload_small = resize_to(payload_gray, band_shape)

    # Take the original band
    orig_tuple = get_band_tuple(coeffs, target_level)
    orig_band = [orig_tuple[0], orig_tuple[1], orig_tuple[2]][idx]

    # Prepare payload to have comparable energy
    payload_norm = normalize_like(payload_small, orig_band, mode="match_std")

    # Combine
    if mode == "replace":
        new_band = (alpha * payload_norm)
    else:  # 'add'
        new_band = orig_band + alpha * payload_norm

    # Build new tuple and set
    new_tuple = list(orig_tuple)
    new_tuple[idx] = new_band
    set_band_tuple(coeffs, target_level, tuple(new_tuple))

    # Reconstruct
    marked = waverec2_gray(coeffs, wavelet)

    return {
        "marked_gray": marked,
        "original_band": orig_band,
        "payload_used": payload_small,
        "band_after": new_band
    }

def extract_payload_from_marked(marked_gray: np.ndarray,
                                levels: int,
                                wavelet: str,
                                target_level: int,
                                band: str) -> np.ndarray:
    """Extract selected band from the marked image's DWT and scale to [0,1] for viewing."""
    coeffs_m = wavedec2_gray(marked_gray, levels, wavelet)
    (mH, mV, mD) = get_band_tuple(list(coeffs_m), target_level)
    band = band.upper()
    B = {'H': mH, 'V': mV, 'D': mD}[band]
    # Normalize for view
    v = B - float(np.min(B))
    v = v / (float(np.max(v)) + 1e-6)
    return v.astype(np.float32)


# ----------------------- Streamlit UI -----------------------

st.set_page_config(page_title="Wavelet Meta Transport (Layer Swap)", layout="wide")
st.title("Wavelets: Transport Meta Information by Exchanging a Detail Layer")

with st.expander("What this does (quick)"):
    st.markdown(
        "- We decompose the COVER image into wavelet subbands (details at multiple scales/orientations).\n"
        "- We **replace or add** a PAYLOAD into a selected **detail band** (H/V/D at a chosen level).\n"
        "- Inverse wavelet transform gives a **marked image** carrying the payload.\n"
        "- Extracting the same band from the marked image recovers the payload.\n"
        "- This is a simple, visual form of *wavelet-domain steganography* / metadata transport."
    )

left, right = st.columns([0.9, 1.1])

with left:
    st.subheader("Inputs")

    cover_file = st.file_uploader("Upload COVER image", type=["png","jpg","jpeg","bmp"])
    payload_mode = st.radio("Payload source", ["Upload image", "Text"], horizontal=True)
    payload_file = None
    payload_text = None
    if payload_mode == "Upload image":
        payload_file = st.file_uploader("Upload PAYLOAD image", type=["png","jpg","jpeg","bmp"], key="payload")
    else:
        payload_text = st.text_input("Enter short payload text", value="Hello Wavelets!")

    use_random_cover = st.checkbox("Use random COVER (ignore upload if checked)", value=False)

    st.markdown("**Wavelet parameters**")
    levels = st.slider("MRA levels", 1, 5, 3)
    wavelet = st.text_input("Wavelet name", "db2")
    target_level = st.slider("Target level (1=fine, L=coarse)", 1, levels, min(2, levels))
    band = st.selectbox("Detail band", ["H", "V", "D"])
    mode = st.selectbox("Embed mode", ["add", "replace"])
    alpha = st.slider("Strength (alpha)", 0.05, 2.0, 0.5, step=0.05)

    run = st.button("Run")

with right:
    st.subheader("Results (side-by-side)")

    if run:
        # --- Prepare COVER ---
        if use_random_cover or (cover_file is None and not use_random_cover):
            # Random cover for demo
            rnd = (np.random.rand(360, 480, 3) * 255).astype(np.uint8)
            if HAVE_CV2:
                cv2.putText(rnd, "COVER", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
            cover_bgr = rnd
        else:
            data = cover_file.read()
            arr = np.frombuffer(data, np.uint8)
            if HAVE_CV2:
                cover_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                cover_bgr = np.array(Image.open(io.BytesIO(data)).convert("RGB"))[..., ::-1]

        cover_gray = to_gray(cover_bgr)

        # --- Prepare PAYLOAD ---
        if payload_mode == "Upload image" and payload_file is not None:
            pdata = payload_file.read()
            parr = np.frombuffer(pdata, np.uint8)
            if HAVE_CV2:
                pay_bgr = cv2.imdecode(parr, cv2.IMREAD_COLOR)
                payload_gray = to_gray(pay_bgr)
            else:
                payload_gray = to_gray(np.array(Image.open(io.BytesIO(pdata)).convert("RGB")))
        else:
            payload_gray = text_to_image(payload_text or "Wavelets!", size=(128,128))

        # --- Embed ---
        try:
            out = embed_payload_in_band(
                cover_gray, payload_gray,
                levels=levels, wavelet=wavelet,
                target_level=target_level, band=band,
                mode=mode, alpha=alpha
            )
        except Exception as e:
            st.error(f"Wavelet error: {e}")
            st.stop()

        marked_gray = out["marked_gray"]
        original_band = out["original_band"]
        payload_used = out["payload_used"]
        band_after = out["band_after"]

        # Extract for verification
        extracted = extract_payload_from_marked(
            marked_gray, levels=levels, wavelet=wavelet,
            target_level=target_level, band=band
        )

        # Recon to RGB for display
        cover_rgb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2RGB) if HAVE_CV2 else cover_bgr[..., ::-1]
        marked_rgb = to_uint8(np.stack([marked_gray]*3, axis=-1))

        # Metrics
        p = psnr(cover_gray, marked_gray)

        # Visual grid
        row1 = [
            (cover_rgb, "COVER (RGB)"),
            (to_uint8(np.stack([cover_gray]*3, -1)), "COVER (gray)"),
            (to_uint8(np.stack([payload_gray]*3, -1)), "PAYLOAD (input)")
        ]
        grid_show(row1, cols=3)

        row2 = [
            (to_uint8(np.stack([payload_used]*3, -1)), f"PAYLOAD resized → band [{band}]@L{target_level}"),
            (to_uint8(np.stack([marked_gray]*3, -1)), f"MARKED (PSNR vs cover: {p:.2f} dB)"),
            (to_uint8(np.stack([heatmap_abs_diff(cover_gray, marked_gray)]*3, -1)), "Δ |cover - marked| heatmap")
        ]
        grid_show(row2, cols=3)

        # Show the band before/after (normalized for viewing)
        def norm01(x):
            y = x - x.min()
            y = y / (y.max() + 1e-6)
            return y

        band_vis = norm01(original_band)
        band_after_vis = norm01(band_after)
        row3 = [
            (to_uint8(np.stack([band_vis]*3, -1)), f"Original band [{band}]@L{target_level} (view)"),
            (to_uint8(np.stack([band_after_vis]*3, -1)), f"Band after embed (view)"),
            (to_uint8(np.stack([extracted]*3, -1)), f"Extracted band [{band}]@L{target_level} (from MARKED)")
        ]
        grid_show(row3, cols=3)

        with st.expander("What to look for (explanation)"):
            st.markdown(
                f"- We embedded the payload into **{band}-detail at level {target_level}** using **{mode}** mode "
                f"with strength **alpha={alpha:.2f}**.\n"
                "- The marked image should look very similar to the cover (see PSNR).\n"
                "- The **Extracted band** should visually resemble the payload (after normalization) "
                "since we read back the very same subband from the marked image.\n"
                "- Higher levels (coarser) and smaller alpha usually give less visible changes; "
                "lower levels and larger alpha make the payload more apparent but lower PSNR."
            )

    else:
        st.info("Upload a COVER and a PAYLOAD (or type text), pick parameters, then press **Run**.")
