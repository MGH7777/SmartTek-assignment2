# app.py

import io
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import streamlit as st

# Core libs
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

# DICOM 
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    HAVE_DICOM = True
except Exception:
    HAVE_DICOM = False


# ---------------------- Utilities ----------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """Ensure grayscale float32 in [0,1]."""
    if img.ndim == 3 and img.shape[2] == 3:
        if HAVE_CV2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = (0.299*img[...,0] + 0.587*img[...,1] + 0.114*img[...,2]).astype(np.float32)
        gray = gray.astype(np.float32)
    elif img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")
    if gray.max() > 1.5:
        gray = gray / 255.0
    return np.clip(gray, 0.0, 1.0).astype(np.float32)

def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

def ensure_three_channel(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    return gray

def resize_to(img: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize image to target (H,W)."""
    h, w = target_shape
    if HAVE_CV2:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    y_idx = (np.linspace(0, img.shape[0]-1, h)).astype(int)
    x_idx = (np.linspace(0, img.shape[1]-1, w)).astype(int)
    return img[np.ix_(y_idx, x_idx)]

def grid_show(items: List[Tuple[np.ndarray, str]], cols: int = 3):
    """
    Display a list of (image, caption) in a grid with given number of columns.
    Automatically wraps to multiple rows as needed.
    """
    if not items:
        return
    chunks = [items[i:i+cols] for i in range(0, len(items), cols)]
    for chunk in chunks:
        col_objs = st.columns(len(chunk), gap="small")
        for (img, cap), col in zip(chunk, col_objs):
            with col:
                col.image(img, caption=cap, clamp=True, use_container_width=True)


# ---------------------- DICOM helpers ----------------------

def _dicom_to_float(image: np.ndarray, ds) -> np.ndarray:
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = image.astype(np.float32) * slope + intercept
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr[:] = 0.0
    return arr

def dicom_bytes_to_frames_rgb(dcm_bytes: bytes) -> List[np.ndarray]:
    """Return list of RGB frames from a DICOM (single or multiframe)."""
    if not HAVE_DICOM:
        raise RuntimeError("pydicom is not available")
    ds = pydicom.dcmread(io.BytesIO(dcm_bytes), force=True)
    arr = ds.pixel_array  # handled by pylibjpeg if compressed

    frames = []
    if arr.ndim == 3 and getattr(ds, "NumberOfFrames", 1) > 1 and arr.shape[0] == int(ds.NumberOfFrames):
        iterable = list(arr)
    else:
        iterable = [arr]

    for frame in iterable:
        if frame.ndim == 2:
            f = _dicom_to_float(frame, ds)
            rgb = np.stack([f, f, f], axis=-1)
        elif frame.ndim == 3:
            f = frame.astype(np.float32)
            for c in range(f.shape[2]):
                cmin, cmax = float(f[..., c].min()), float(f[..., c].max())
                f[..., c] = 0.0 if cmax == cmin else (f[..., c] - cmin) / (cmax - cmin)
            rgb = f
        else:
            frame = np.squeeze(frame)
            f = _dicom_to_float(frame, ds)
            rgb = np.stack([f, f, f], axis=-1)

        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        frames.append(rgb8)
    return frames


# ---------------------- MRA (Wavelet + fallback) ----------------------

def mra_wavelet(gray: np.ndarray, levels: int = 3, wavelet: str = 'db2') -> Dict[str, Any]:
    """
    Compute MRA approximations and detail magnitudes.
    Uses PyWavelets if available, else Gaussian/Laplacian pyramid fallback.
    """
    H, W = gray.shape[:2]
    results = {"approximations": [], "details": [], "method": "wavelet" if HAVE_PYWT else "gaussian_fallback"}

    if HAVE_PYWT:
        approx_img = gray.copy()
        approx_levels = []
        for _ in range(levels):
            cA, (cH, cV, cD) = pywt.dwt2(approx_img, wavelet=wavelet)
            approx_levels.append(cA)
            approx_img = cA
        for cA in approx_levels:
            results["approximations"].append(resize_to(cA, (H, W)))

        approx_img = gray.copy()
        for _ in range(levels):
            cA, (cH, cV, cD) = pywt.dwt2(approx_img, wavelet=wavelet)
            mag = np.sqrt(cH**2 + cV**2 + cD**2)
            mag = mag / (mag.max() + 1e-6)
            results["details"].append(resize_to(mag, (H, W)))
            approx_img = cA
        return results

    # Fallback pyramid
    current = gray.copy()
    for _ in range(levels):
        if HAVE_CV2:
            down = cv2.pyrDown(current)
            up = resize_to(down, current.shape[:2])
            detail = np.abs(current - up)
        else:
            kernel = np.ones((5,5), dtype=np.float32) / 25.0
            padded = np.pad(current, 2, mode='reflect')
            blur = np.zeros_like(current)
            for i in range(current.shape[0]):
                for j in range(current.shape[1]):
                    blur[i, j] = np.sum(padded[i:i+5, j:j+5] * kernel)
            down = blur[::2, ::2]
            up = resize_to(down, current.shape[:2])
            detail = np.abs(current - up)
        results["approximations"].append(resize_to(down, (H, W)))
        results["details"].append(detail / (detail.max() + 1e-6))
        current = down
    return results


# ---------------------- Edges, Morphology, Features ----------------------

def detect_edges(gray: np.ndarray,
                 method: str = "canny",
                 low: int = 100,
                 high: int = 200) -> np.ndarray:
    """
    Edge detection with adjustable thresholds for Canny.
    `low` and `high` are used only when method == 'canny'.
    """
    g8 = to_uint8(gray)
    m = method.lower()
    if m == "canny" and HAVE_CV2:
        e = cv2.Canny(g8, int(low), int(high)).astype(np.float32) / 255.0
        return e
    elif m == "sobel":
        if HAVE_CV2:
            gx = cv2.Sobel(g8, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(g8, cv2.CV_32F, 0, 1, ksize=3)
        else:
            gx = np.zeros_like(gray, dtype=np.float32); gy = np.zeros_like(gray, dtype=np.float32)
            gx[:, 1:] = gray[:, 1:] - gray[:, :-1]; gy[1:, :] = gray[1:, :] - gray[:-1, :]
        mag = np.sqrt(gx**2 + gy**2)
        return mag / (mag.max() + 1e-6)
    else:  # Laplacian
        if HAVE_CV2:
            lap = cv2.Laplacian(g8, cv2.CV_32F, ksize=3)
        else:
            k = np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32)
            padded = np.pad(gray, 1, mode='reflect')
            lap = np.zeros_like(gray, dtype=np.float32)
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    lap[i, j] = np.sum(padded[i:i+3, j:j+3] * k)
        lap = np.abs(lap)
        return lap / (lap.max() + 1e-6)

def morph_process(edge_img: np.ndarray, op: str = "close", k: int = 3) -> np.ndarray:
    k = max(1, int(k))
    if HAVE_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        e8 = to_uint8(edge_img)
        if op == "open":
            m = cv2.morphologyEx(e8, cv2.MORPH_OPEN, kernel)
        elif op == "erode":
            m = cv2.erode(e8, kernel, iterations=1)
        elif op == "dilate":
            m = cv2.dilate(e8, kernel, iterations=1)
        else:
            m = cv2.morphologyEx(e8, cv2.MORPH_CLOSE, kernel)
        return m.astype(np.float32) / 255.0
    # simple fallback
    return (edge_img > 0.3).astype(np.float32)

def feature_extract(gray: np.ndarray, prefer_sift: bool = True) -> Tuple[Any, Any, np.ndarray]:
    vis = ensure_three_channel(gray)
    vis8 = to_uint8(vis)
    if HAVE_CV2:
        sift = None
        if prefer_sift:
            try:
                sift = cv2.SIFT_create(nfeatures=400)
            except Exception:
                sift = None
        if sift is not None:
            kps, desc = sift.detectAndCompute(to_uint8(gray), None)
            kp_img = cv2.drawKeypoints(vis8, kps, None)
            return kps, desc, cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)
        # ORB fallback
        orb = cv2.ORB_create(nfeatures=400)
        kps, desc = orb.detectAndCompute(to_uint8(gray), None)
        kp_img = cv2.drawKeypoints(vis8, kps, None)
        return kps, desc, cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB)
    return [], None, vis8

def extract_rois_from_edges(edge_img: np.ndarray, min_area: int = 200) -> Tuple[List[Tuple[int,int,int,int]], np.ndarray]:
    h, w = edge_img.shape[:2]
    e8 = to_uint8(edge_img)
    boxes = []
    if HAVE_CV2:
        contours, _ = cv2.findContours(e8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh >= min_area:
                boxes.append((x, y, bw, bh))
                cv2.rectangle(vis, (x, y), (x+bw, y+bh), (255,255,255), 2)
        return boxes, cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    # fallback: raw edge stack
    return boxes, np.stack([e8]*3, axis=-1)


# ---------------------- Image & Video Processing ----------------------

def process_image(img_bgr: np.ndarray,
                  levels: int,
                  wavelet: str,
                  edge_method: str,
                  morph_op: str,
                  morph_k: int,
                  prefer_sift: bool,
                  canny_low: int,
                  canny_high: int) -> Dict[str, Any]:
    gray = to_gray(img_bgr)
    mra = mra_wavelet(gray, levels=levels, wavelet=wavelet)
    approx_for_edges = mra["approximations"][-1] if mra["approximations"] else gray
    edges = detect_edges(approx_for_edges, method=edge_method, low=canny_low, high=canny_high)
    morph = morph_process(edges, op=morph_op, k=morph_k)
    kps, desc, kp_img = feature_extract(gray, prefer_sift=prefer_sift)
    boxes, roi_vis = extract_rois_from_edges(morph, min_area=max(300, (img_bgr.shape[0]*img_bgr.shape[1])//250))
    return {
        "orig_rgb": cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if HAVE_CV2 else img_bgr[..., ::-1],
        "gray": gray,
        "mra": mra,
        "edges": edges,
        "morph": morph,
        "keypoints_image": kp_img,
        "roi_boxes": boxes,
        "roi_overlay": roi_vis,
    }

def process_video_frames(frames_bgr: List[np.ndarray],
                         levels: int,
                         wavelet: str,
                         mode: str,          
                         edge_method: str,
                         morph_op: str,
                         morph_k: int,
                         canny_low: int,
                         canny_high: int) -> List[np.ndarray]:
    """Return list of processed RGB frames (only visualization; no file IO)."""
    processed = []
    prev_gray = None
    if not frames_bgr:
        return processed
    H, W = frames_bgr[0].shape[:2]
    for f in frames_bgr:
        gray = to_gray(f)
        if mode == "diff" and prev_gray is not None:
            inp = np.abs(gray - prev_gray)
        else:
            inp = gray
        mra = mra_wavelet(inp, levels=levels, wavelet=wavelet)
        approx = mra["approximations"][-1] if mra["approximations"] else inp
        edges = detect_edges(approx, method=edge_method, low=canny_low, high=canny_high)
        morph = morph_process(edges, op=morph_op, k=morph_k)
        boxes, _ = extract_rois_from_edges(morph, min_area=max(150, (H*W)//400))
        canvas = f.copy()
        if HAVE_CV2:
            for (x,y,bw,bh) in boxes:
                cv2.rectangle(canvas, (x,y), (x+bw, y+bh), (255,255,255), 2)
        processed.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB) if HAVE_CV2 else canvas[..., ::-1])
        prev_gray = gray
    return processed


# ---------------------- Streamlit GUI ----------------------

st.set_page_config(page_title="Wavelet MRA GUI", layout="wide")
st.title("Wavelet Multi-Resolution Analysis (MRA) — Edges • Morphology • SIFT/ORB • Video")


left, right = st.columns([0.9, 1.1])

with left:
    st.subheader("Inputs")
    file = st.file_uploader("Upload an image, DICOM, or a video",
                            type=["png","jpg","jpeg","bmp","dcm","mp4","avi","mov","mkv"])
    use_random = st.checkbox("Use random image (ignore upload if checked)", value=False)

    st.markdown("**MRA & processing parameters**")
    levels = st.slider("MRA levels", 1, 5, 3)
    wavelet = st.text_input("Wavelet (PyWavelets)", "db2")
    edge_method = st.selectbox("Edge detector", ["canny", "sobel", "laplacian"])

    if edge_method.lower() == "canny":
        st.markdown("**Canny Thresholds**")
        canny_low = st.slider("Canny low", 0, 255, 100)
        canny_high = st.slider("Canny high", 0, 255, 200)
    else:
        canny_low, canny_high = 100, 200  # defaults

    morph_op = st.selectbox("Morphology", ["close", "open", "erode", "dilate"])
    morph_k = st.slider("Morph kernel", 1, 15, 3, step=2)
    prefer_sift = st.checkbox("Prefer SIFT (falls back to ORB)", value=True)

    st.markdown("**Video mode (only used for videos / multiframe DICOM)**")
    video_mode = st.radio("Frame mode", ["intra", "diff"], horizontal=True)

    run = st.button("Run")

with right:
    st.subheader("Results")

    if run:
        results_grid: List[Tuple[np.ndarray, str]] = []

        # Random image path
        if use_random:
            rnd = (np.random.rand(320, 480) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(rnd, cv2.COLOR_GRAY2BGR) if HAVE_CV2 else np.stack([rnd, rnd, rnd], axis=-1)
            out = process_image(img_bgr, levels, wavelet, edge_method, morph_op, morph_k, prefer_sift, canny_low, canny_high)

            # Row 1
            grid_show([
                (out["orig_rgb"], "Original"),
                (out["gray"], "Grayscale"),
                (out["mra"]["approximations"][0], "MRA Approx L1"),
            ], cols=3)

            # Row 2 (more approximations & details)
            row2 = []
            if len(out["mra"]["approximations"]) > 1:
                row2.append((out["mra"]["approximations"][1], "MRA Approx L2"))
            if len(out["mra"]["approximations"]) > 2:
                row2.append((out["mra"]["approximations"][2], "MRA Approx L3"))
            if out["mra"]["details"]:
                row2.append((out["mra"]["details"][0], "MRA Detail L1"))
            grid_show(row2, cols=3)

            # Row 3 (remaining details)
            row3 = []
            if len(out["mra"]["details"]) > 1:
                row3.append((out["mra"]["details"][1], "MRA Detail L2"))
            if len(out["mra"]["details"]) > 2:
                row3.append((out["mra"]["details"][2], "MRA Detail L3"))
            row3.append((out["edges"], "Edges"))
            grid_show(row3, cols=3)

            # Row 4 (morph, keypoints, rois)
            grid_show([
                (out["morph"], "Morphology"),
                (out["keypoints_image"], "Feature Keypoints"),
                (out["roi_overlay"], "ROIs (from edges)"),
            ], cols=3)

        # Uploaded file path
        elif file is not None:
            suffix = file.name.split(".")[-1].lower()

            # --- Images (PNG/JPG/BMP) ---
            if suffix in ["png","jpg","jpeg","bmp"]:
                if not HAVE_CV2:
                    st.error("OpenCV is required to decode images.")
                else:
                    data = file.read()
                    arr = np.frombuffer(data, np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    out = process_image(img_bgr, levels, wavelet, edge_method, morph_op, morph_k, prefer_sift, canny_low, canny_high)

                    grid_show([
                        (out["orig_rgb"], "Original"),
                        (out["gray"], "Grayscale"),
                        (out["mra"]["approximations"][0], "MRA Approx L1"),
                    ], cols=3)

                    row2 = []
                    if len(out["mra"]["approximations"]) > 1:
                        row2.append((out["mra"]["approximations"][1], "MRA Approx L2"))
                    if len(out["mra"]["approximations"]) > 2:
                        row2.append((out["mra"]["approximations"][2], "MRA Approx L3"))
                    if out["mra"]["details"]:
                        row2.append((out["mra"]["details"][0], "MRA Detail L1"))
                    grid_show(row2, cols=3)

                    row3 = []
                    if len(out["mra"]["details"]) > 1:
                        row3.append((out["mra"]["details"][1], "MRA Detail L2"))
                    if len(out["mra"]["details"]) > 2:
                        row3.append((out["mra"]["details"][2], "MRA Detail L3"))
                    row3.append((out["edges"], "Edges"))
                    grid_show(row3, cols=3)

                    grid_show([
                        (out["morph"], "Morphology"),
                        (out["keypoints_image"], "Feature Keypoints"),
                        (out["roi_overlay"], "ROIs (from edges)"),
                    ], cols=3)

            # --- DICOM (.dcm) ---
            elif suffix == "dcm":
                if not HAVE_DICOM:
                    st.error("pydicom is not available in this environment.")
                else:
                    data = file.read()
                    frames_rgb = dicom_bytes_to_frames_rgb(data)

                    if len(frames_rgb) == 1:
                        img_rgb = frames_rgb[0]
                        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR) if HAVE_CV2 else img_rgb[..., ::-1]
                        out = process_image(img_bgr, levels, wavelet, edge_method, morph_op, morph_k, prefer_sift, canny_low, canny_high)

                        grid_show([
                            (out["orig_rgb"], "Original"),
                            (out["gray"], "Grayscale"),
                            (out["mra"]["approximations"][0], "MRA Approx L1"),
                        ], cols=3)

                        row2 = []
                        if len(out["mra"]["approximations"]) > 1:
                            row2.append((out["mra"]["approximations"][1], "MRA Approx L2"))
                        if len(out["mra"]["approximations"]) > 2:
                            row2.append((out["mra"]["approximations"][2], "MRA Approx L3"))
                        if out["mra"]["details"]:
                            row2.append((out["mra"]["details"][0], "MRA Detail L1"))
                        grid_show(row2, cols=3)

                        row3 = []
                        if len(out["mra"]["details"]) > 1:
                            row3.append((out["mra"]["details"][1], "MRA Detail L2"))
                        if len(out["mra"]["details"]) > 2:
                            row3.append((out["mra"]["details"][2], "MRA Detail L3"))
                        row3.append((out["edges"], "Edges"))
                        grid_show(row3, cols=3)

                        grid_show([
                            (out["morph"], "Morphology"),
                            (out["keypoints_image"], "Feature Keypoints"),
                            (out["roi_overlay"], "ROIs (from edges)"),
                        ], cols=3)

                    else:
                        if not HAVE_CV2:
                            st.error("OpenCV is required for video-like processing.")
                        else:
                            frames_bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames_rgb]
                            proc = process_video_frames(frames_bgr, levels, wavelet, video_mode,
                                                        edge_method, morph_op, morph_k, canny_low, canny_high)
                            if proc:
                                # Show three representative frames side-by-side
                                idxs = [0, min(30, len(proc)-1), min(90, len(proc)-1)]
                                grid_show([(proc[i], f"Processed Frame {i}") for i in idxs], cols=3)
                            else:
                                st.warning("No frames processed.")

            # --- Standard videos ---
            else:
                if not HAVE_CV2:
                    st.error("OpenCV is required for video processing.")
                else:
                    tmp_path = f"./_upload_{file.name}"
                    with open(tmp_path, "wb") as f:
                        f.write(file.read())

                    cap = cv2.VideoCapture(tmp_path)
                    frames = []
                    ok, fr = cap.read()
                    max_frames = 180 
                    while ok:
                        frames.append(fr)
                        if len(frames) >= max_frames:
                            break
                        ok, fr = cap.read()
                    cap.release()

                    proc = process_video_frames(frames, levels, wavelet, video_mode,
                                                edge_method, morph_op, morph_k, canny_low, canny_high)
                    if proc:
                        idxs = [0, min(30, len(proc)-1), min(90, len(proc)-1)]
                        grid_show([(proc[i], f"Processed Frame {i}") for i in idxs], cols=3)
                    else:
                        st.warning("No frames processed.")
        else:
            st.info("Upload an image/DICOM/video or check “Use random image”, then click Run.")
    