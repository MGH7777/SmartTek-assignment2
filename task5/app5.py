import numpy as np
import pywt
from PIL import Image, ImageDraw
import streamlit as st

st.set_page_config(page_title="Wavelet meta info", layout="wide")


def load_image_to_gray(uploaded_file):
    """Load uploaded image as grayscale float [0,1]."""
    img = Image.open(uploaded_file).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def text_to_image(text, size=(128, 128)):
    """Very simple text -> small grayscale image."""
    w, h = size[1], size[0]
    img = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(img)
    lines = []
    words = text.split()
    line = ""
    for word in words:
        if len(line) + 1 + len(word) < 12:
            if line == "":
                line = word
            else:
                line = line + " " + word
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = 10
    for ln in lines[:5]:
        draw.text((5, y), ln, fill=255)
        y += 18

    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def dwt2_levels(img_gray, wavelet_name, levels):
    """Compute 2D DWT with several levels."""
    coeffs = pywt.wavedec2(img_gray, wavelet_name, level=levels)
    return coeffs


def idwt2_levels(coeffs, wavelet_name):
    return pywt.waverec2(coeffs, wavelet_name)


def resize_payload_to_band(payload, band):
    """Resize payload image to the size of a detail band."""
    H, W = band.shape
    img = Image.fromarray((payload * 255).astype(np.uint8))
    img = img.resize((W, H), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    return arr


def embed_payload(cover_gray, payload_gray, levels, wavelet_name,
                  target_level, band_name, mode, alpha):
    """
    Put payload into one detail band at one level.
    target_level: 1 = finest, levels = coarsest.
    band_name: "H", "V" or "D".
    mode: "add" or "replace".
    """
    coeffs = dwt2_levels(cover_gray, wavelet_name, levels)
    cA = coeffs[0]
    details = list(coeffs[1:])  

    idx = levels - target_level
    cH, cV, cD = details[idx]

    if band_name == "H":
        band = cH
    elif band_name == "V":
        band = cV
    else:
        band = cD

    payload_small = resize_payload_to_band(payload_gray, band)

    # Normalize payload to have similar mean/std as current band
    b_mean = band.mean()
    b_std = band.std() + 1e-6
    p = (payload_small - payload_small.mean()) / (payload_small.std() + 1e-6)
    p = p * b_std + b_mean

    if mode == "add":
        new_band = band + alpha * p
    else:  
        new_band = (1.0 - alpha) * band + alpha * p

    # Put band back
    if band_name == "H":
        cH = new_band
    elif band_name == "V":
        cV = new_band
    else:
        cD = new_band

    details[idx] = (cH, cV, cD)
    new_coeffs = [cA] + details
    marked = idwt2_levels(new_coeffs, wavelet_name)

    return marked, band, payload_small, new_band


def extract_band(marked_gray, levels, wavelet_name, target_level, band_name):
    coeffs = dwt2_levels(marked_gray, wavelet_name, levels)
    details = list(coeffs[1:])
    idx = levels - target_level
    cH, cV, cD = details[idx]
    if band_name == "H":
        B = cH
    elif band_name == "V":
        B = cV
    else:
        B = cD
    # Normalize to [0,1] to show
    B = B - B.min()
    B = B / (B.max() + 1e-6)
    return B.astype(np.float32)


def psnr(x, y):
    mse = np.mean((x - y) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10 * np.log10(1.0 / mse)


def norm01(a):
    a = a - a.min()
    return a / (a.max() + 1e-6)


def main():
    st.title("Wavelet meta info: embed a payload in one detail band")

    st.write("Upload a **cover image** and either a **payload image** or write some text. "
             "Then choose a wavelet band where the payload is stored.")

    st.sidebar.header("Parameters")
    wavelet_name = st.sidebar.selectbox("Wavelet", ["haar"])
    levels = st.sidebar.slider("Levels", 1, 4, 2)
    target_level = st.sidebar.slider("Target level (1=finest)", 1, levels, 1)
    band_name = st.sidebar.selectbox("Band", ["H", "V", "D"])
    mode = st.sidebar.selectbox("Mode", ["add", "replace"])
    alpha = st.sidebar.slider("Alpha (strength)", 0.0, 1.0, 0.3, 0.05)

    cover_file = st.file_uploader("Cover image", type=["png", "jpg", "jpeg"])
    payload_file = st.file_uploader("Payload image (optional)", type=["png", "jpg", "jpeg"])
    payload_text = st.text_input("Or write payload text (used if no payload image):", "Hello wavelets")

    if st.button("Run") and cover_file is not None:
        cover_gray = load_image_to_gray(cover_file)

        if payload_file is not None:
            payload_gray = load_image_to_gray(payload_file)
        else:
            payload_gray = text_to_image(payload_text)

        marked_gray, orig_band, payload_small, new_band = embed_payload(
            cover_gray, payload_gray, levels, wavelet_name,
            target_level, band_name, mode, alpha
        )

        extracted = extract_band(marked_gray, levels, wavelet_name, target_level, band_name)

        # Clip and normalize for safe display
        cover_show = np.clip(cover_gray, 0, 1)
        marked_show = np.clip(marked_gray, 0, 1)
        diff = np.abs(marked_show - cover_show)
        diff = diff / (diff.max() + 1e-6)

        # ---------- IMAGE LAYOUT ----------

        st.subheader("Images")
        c1, c2 = st.columns(2)
        with c1:
            st.image(cover_show, caption="Cover (gray)", clamp=True, use_container_width=True)
        with c2:
            st.image(payload_gray, caption="Original payload (gray)", clamp=True, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.image(marked_show, caption="Marked image (after embedding)", clamp=True, use_container_width=True)
        with c4:
            st.image(diff, caption="Absolute difference |marked - cover|", clamp=True, use_container_width=True)

        st.image(extracted, caption="Extracted band (normalized)", clamp=True, use_container_width=True)

        st.subheader("Bands at chosen level")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.image(norm01(orig_band), caption="Band before embedding", clamp=True, use_container_width=True)
        with b2:
            st.image(norm01(new_band), caption="Band after embedding", clamp=True, use_container_width=True)
        with b3:
            st.image(norm01(payload_small), caption="Payload resized to band size", clamp=True, use_container_width=True)

        value_psnr = psnr(cover_show, marked_show)
        st.write(f"PSNR between cover and marked: **{value_psnr:.2f} dB**")

    else:
        st.info("Choose images and parameters, then press Run.")


if __name__ == "__main__":
    main()
