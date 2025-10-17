# streamlit_main.py
import io, zipfile, json
import numpy as np
import streamlit as st

# core functions from app1.py
from app1 import (
    compress_dct, decompress_dct,
    compress_fft, decompress_fft,
    psnr, ssim, to_uint8, load_first_dicom_from_zip
)

try:
    import pydicom
    HAVE_PYDICOM = True
except Exception:
    HAVE_PYDICOM = False

st.set_page_config(page_title="Lossy Compression: FFT vs DCT", layout="wide")
st.title("Lossy Compression Pipeline (FFT vs DCT) — Python")

st.markdown("""
Upload a **DICOM** file (`.dcm`) or a **ZIP** containing one or more DICOMs. 
Choose **DCT (JPEG-like 8×8)** or **FFT (global)** compression, tweak the quality, and compare reconstruction.
""")

colL, colR = st.columns([1, 1])

with st.sidebar:
    st.header("Compression Settings")
    method = st.selectbox("Method", ["DCT (8×8 block)", "FFT (global)"])
    quality = st.slider("Quality (higher = better)", min_value=1, max_value=100, value=60, step=1)
    use_huffman = st.checkbox("Huffman coding (entropy stage)", value=True)
    st.caption("Tip: lower quality = stronger quantization = higher compression but more artifacts.")

    st.header("Input")
    uploaded = st.file_uploader("Upload .dcm or .zip", type=["dcm", "zip"])

    st.header("Notes")
    if not HAVE_PYDICOM:
        st.error("`pydicom` is not installed. Please install it in your active environment.")

# Load image
img8 = None
orig_bits_per_px = 8
filename = None

if uploaded is not None and HAVE_PYDICOM:
    data = uploaded.read()
    if uploaded.name.lower().endswith(".zip"):
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
            arr, bits, name = load_first_dicom_from_zip(zf)
            filename = name
        except Exception as e:
            st.error(f"Failed to read ZIP: {e}")
            arr = None
            bits = 8
    else:
        try:
            ds = pydicom.dcmread(io.BytesIO(data))
            arr = ds.pixel_array
            bits = int(getattr(ds, "BitsStored", 16))
            filename = uploaded.name
        except Exception as e:
            st.error(f"Failed to read DICOM: {e}")
            arr = None
            bits = 8
    if arr is not None:
        img8 = to_uint8(arr)
        orig_bits_per_px = bits

if img8 is not None:
    colL.subheader("Original")
    colL.image(img8, clamp=True, width='stretch')
    
    if st.button("Run Compression / Reconstruction", type="primary"):
        with st.spinner("Compressing..."):
            try:
                if method.startswith("DCT"):
                    byts, meta = compress_dct(img8, quality=quality, use_huffman=use_huffman, bs=8)
                    rec = decompress_dct(byts, meta)
                else:
                    byts, meta = compress_fft(img8, quality=quality, use_huffman=use_huffman)
                    rec = decompress_fft(byts, meta)

                colR.subheader("Reconstructed")
                colR.image(rec,  clamp=True, width='stretch')
                
                st.markdown("### Debug Info")
                st.write(f"Original image range: {img8.min()} - {img8.max()}")
                st.write(f"Reconstructed image range: {rec.min()} - {rec.max()}")
                st.write(f"Original shape: {img8.shape}, Reconstructed shape: {rec.shape}")

                # Metrics
                m = float(np.mean((img8.astype(np.float64) - rec.astype(np.float64)) ** 2))
                ps = psnr(img8, rec)
                ss = ssim(img8, rec)
                st.markdown("### Quality Metrics")
                mcol1, mcol2, mcol3, mcol4 = st.columns([1, 1, 1, 1])
                with mcol1:
                    st.metric("MSE", f"{m:.2f}")
                with mcol2:
                    st.metric("PSNR (dB)", f"{ps:.2f}")
                with mcol3:
                    st.metric("SSIM", f"{ss:.4f}")
                # Rough compression ratio estimate
                comp_bits = meta["huff_meta"]["bitlen"] if use_huffman else len(byts) * 8
                orig_bits = img8.size * orig_bits_per_px
                cr = orig_bits / max(1, comp_bits)
                with mcol4:
                    st.metric("Compression Ratio", f"{cr:.2f}×")

                # Download compressed package
                payload = {
                    "meta": meta,
                    "data": byts.hex(),
                    "method": "DCT" if method.startswith("DCT") else "FFT",
                }
                st.download_button(
                    "Download compressed package (.json)",
                    data=json.dumps(payload).encode("utf-8"),
                    file_name=f"compressed_{'dct' if method.startswith('DCT') else 'fft'}.json",
                    mime="application/json",
                )

                # Rebuild from uploaded package
                st.markdown("---")
                st.subheader("Reconstruct from a compressed package")
                pkg = st.file_uploader("Upload a previously downloaded .json package", type=["json"], key="pkg")
                if pkg is not None:
                    try:
                        obj = json.loads(pkg.read().decode("utf-8"))
                        byts2 = bytes.fromhex(obj["data"])
                        meta2 = obj["meta"]
                        method2 = obj.get("method", "DCT")
                        if method2 == "DCT":
                            rec2 = decompress_dct(byts2, meta2)
                        else:
                            rec2 = decompress_fft(byts2, meta2)
                        st.image(rec2, clamp=True, caption="Reconstructed from package", width='stretch')

                    except Exception as e:
                        st.error(f"Failed to reconstruct from package: {e}")
                        
            except Exception as e:
                st.error(f"Compression/Reconstruction failed: {e}")
                st.exception(e)
else:
    st.info("Upload a DICOM (.dcm) file or a ZIP with DICOMs to begin.")
    
    
    
    
    