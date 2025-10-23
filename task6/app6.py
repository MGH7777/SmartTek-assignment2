#!/usr/bin/env python3
"""
Simple adaptive video codec demo:
- I frames (raw)
- D frames (sparse deltas above threshold)
- P frames (block motion estimation + residual)

Usage examples:
  # two images
  python app6.py --frame1 data/f1.png --frame2 data/f2.png --visualize

  # a video (process N consecutive pairs)
  python app6.py --video data/clip.mp4 --pairs 5 --every 1 --visualize

Dependencies (match your environment.yml):
  numpy, opencv-python, matplotlib
"""

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utility helpers
# -----------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

def pad_to_multiple(img: np.ndarray, m: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img.shape[:2]
    ph = (m - (h % m)) % m
    pw = (m - (w % m)) % m
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return img, (ph, pw)

def unpad(img: np.ndarray, pad: Tuple[int, int]) -> np.ndarray:
    ph, pw = pad
    if ph or pw:
        return img[:-ph or None, :-pw or None]
    return img

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    if m == 0:
        return 99.0
    return 10 * np.log10(255.0**2 / m)

# -----------------------------
# Differential / statistics
# -----------------------------

@dataclass
class DiffStats:
    mean_abs: float
    perc_over_thr: float
    thr: int

def absolute_diff_stats(prev: np.ndarray, curr: np.ndarray, thr: int) -> Tuple[np.ndarray, np.ndarray, DiffStats]:
    diff = cv2.absdiff(curr, prev)
    mask = (diff >= thr).astype(np.uint8)
    mean_abs = float(np.mean(diff))
    perc = float(np.mean(mask) * 100.0)
    return diff, mask, DiffStats(mean_abs=mean_abs, perc_over_thr=perc, thr=thr)

def auto_threshold(prev: np.ndarray, curr: np.ndarray, base: int = 10) -> int:
    """
    Adaptive threshold: base + 0.5 * MAD(diff)
    Keeps noise down but reacts to motion.
    """
    d = cv2.absdiff(curr, prev).astype(np.float32)
    mad = np.median(np.abs(d - np.median(d)))
    t = int(np.clip(base + 0.5 * mad, 4, 64))
    return t

# -----------------------------
# Block matching (full search)
# -----------------------------

def block_match_full_search(
    prev: np.ndarray,
    curr: np.ndarray,
    block: int = 16,
    search: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      mv: H_blocks x W_blocks x 2 array of motion vectors (dy, dx)
      pred: motion-compensated prediction of 'curr' using 'prev'
    """
    prev_f = prev.astype(np.int16)
    curr_f = curr.astype(np.int16)

    H, W = curr.shape
    Hc, Wc = H // block, W // block
    mv = np.zeros((Hc, Wc, 2), dtype=np.int16)
    pred = np.zeros_like(curr)

    # pad previous for boundary-safe search
    prev_pad = cv2.copyMakeBorder(prev_f, search, search, search, search, cv2.BORDER_REPLICATE)

    for by in range(Hc):
        for bx in range(Wc):
            y = by * block
            x = bx * block
            ref_block = curr_f[y:y+block, x:x+block]

            # search window coordinates in prev_pad
            y0 = y + search
            x0 = x + search

            best_sad = 1e18
            best_dy = 0
            best_dx = 0

            for dy in range(-search, search + 1):
                for dx in range(-search, search + 1):
                    cand = prev_pad[y0+dy:y0+dy+block, x0+dx:x0+dx+block]
                    sad = np.sum(np.abs(ref_block - cand))
                    if sad < best_sad:
                        best_sad = sad
                        best_dy, best_dx = dy, dx

            mv[by, bx] = (best_dy, best_dx)
            pred[y:y+block, x:x+block] = prev_pad[y0+best_dy:y0+best_dy+block, x0+best_dx:x0+best_dx+block].astype(np.uint8)

    return mv, pred

def draw_motion_vectors(ax, mv: np.ndarray, block: int):
    Hc, Wc, _ = mv.shape
    Y, X = np.mgrid[0:Hc, 0:Wc]
    U = mv[:, :, 1]  # dx
    V = mv[:, :, 0]  # dy
    ax.quiver(X * block + block // 2, Y * block + block // 2, U, V, angles='xy', scale_units='xy', scale=1)
    ax.set_title("Motion Vectors (P-frame)")
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlim(0, Wc * block)
    ax.set_ylim(Hc * block, 0)

# -----------------------------
# Simple codec (toy)
# -----------------------------

@dataclass
class EncodedFrame:
    ftype: str  # 'I', 'D', 'P'
    payload: Dict

def encode_adaptive(prev: Optional[np.ndarray], curr: np.ndarray, args) -> Tuple[EncodedFrame, np.ndarray, Dict]:
    """
    Returns:
      encoded, reconstructed(gray), diagnostics
    """
    if prev is None:
        # I-frame: store raw
        rec = curr.copy()
        payload = {"frame": curr.copy()}
        return EncodedFrame("I", payload), rec, {"psnr": 99.0}

    # decide threshold
    thr = args.threshold if args.threshold is not None else auto_threshold(prev, curr, base=args.base_threshold)
    diff, mask, st = absolute_diff_stats(prev, curr, thr)

    # simple decision logic:
    # - very still => D-frame (sparse deltas)
    # - otherwise => try P-frame (motion comp) and keep it if residual is small; else D as fallback
    if st.perc_over_thr < args.delta_ratio:  # % pixels over threshold
        # D-frame: store only significant deltas
        coords = np.argwhere(mask > 0)
        values = curr[mask > 0]
        payload = {"coords": coords.astype(np.int16), "values": values.astype(np.uint8), "shape": curr.shape}
        rec = prev.copy()
        rec[mask > 0] = values
        return EncodedFrame("D", payload), rec, {"thr": thr, "over_thr_%": st.perc_over_thr, "mean_abs": st.mean_abs, "psnr": psnr(curr, rec)}

    # P-frame: block matching + residual
    padded_prev, pad = pad_to_multiple(prev, args.block)
    padded_curr, _ = pad_to_multiple(curr, args.block)
    mv, pred = block_match_full_search(padded_prev, padded_curr, block=args.block, search=args.search)
    pred = unpad(pred, pad)

    residual = (curr.astype(np.int16) - pred.astype(np.int16))
    # only keep residual values above thr to be a bit sparse
    rmask = (np.abs(residual) >= thr).astype(np.uint8)
    r_coords = np.argwhere(rmask > 0)
    r_values = (residual[rmask > 0] + 256) % 256  # store as uint8 with wrap
    rec = pred.copy()
    rec[rmask > 0] = (pred[rmask > 0].astype(np.int16) + ((r_values.astype(np.int16) + 128) % 256) - 128).clip(0, 255).astype(np.uint8)

    payload = {
        "mv": mv, "block": args.block, "search": args.search,
        "res_coords": r_coords.astype(np.int16), "res_values": r_values.astype(np.uint8),
        "shape": curr.shape, "pad": pad, "thr": thr
    }
    return EncodedFrame("P", payload), rec, {
        "thr": thr, "residual_sparsity_%": float(np.mean(rmask) * 100.0),
        "psnr": psnr(curr, rec)
    }

# -----------------------------
# IO & Visualization
# -----------------------------

def read_two_frames_from_video(path: str, every: int = 1, start: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    # jump to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start))
    ok, f1 = cap.read()
    for _ in range(every):
        ok2, _ = cap.read()
        if not ok2:
            break
    ok3, f2 = cap.read()
    cap.release()
    if not ok or not ok3:
        raise RuntimeError("Failed to read two frames.")
    return f1, f2

def visualize(prev_c: Optional[np.ndarray], curr_c: np.ndarray, rec: np.ndarray,
              mv: Optional[np.ndarray], diff: Optional[np.ndarray], mask: Optional[np.ndarray],
              args, title_note: str = ""):
    curr = to_gray(curr_c)
    prev = to_gray(prev_c) if prev_c is not None else None

    cols = 3 if mv is None else 4
    plt.figure(figsize=(4 * cols, 10))

    plt.subplot(2, cols, 1)
    plt.imshow(curr, cmap='gray')
    plt.title("Current frame (gray)")
    plt.axis('off')

    if prev is not None:
        plt.subplot(2, cols, 2)
        plt.imshow(prev, cmap='gray')
        plt.title("Previous frame (gray)")
        plt.axis('off')

        if diff is not None:
            plt.subplot(2, cols, 3)
            plt.imshow(diff, cmap='gray')
            plt.title("Abs diff")
            plt.axis('off')

        if mask is not None:
            plt.subplot(2, cols, 4 if cols == 4 else 3)
            plt.imshow(mask, cmap='gray')
            plt.title(f"Diff mask (thr)")
            plt.axis('off')

    plt.subplot(2, cols, cols + 1)
    plt.imshow(rec, cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    plt.subplot(2, cols, cols + 2)
    if prev is not None:
        res = cv2.absdiff(curr, rec)
        plt.imshow(res, cmap='gray')
        plt.title(f"|curr - rec| (PSNR ~ {psnr(curr, rec):.2f} dB)")
    else:
        plt.text(0.5, 0.5, "I-frame", ha='center')
        plt.axis('off')

    if mv is not None:
        ax = plt.subplot(2, cols, cols + 3)
        draw_motion_vectors(ax, mv, args.block)

    plt.suptitle(f"Adaptive Codec Demo {title_note}", fontsize=12)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Simple adaptive codec with delta frames and block motion estimation.")
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--video", type=str, help="Path to video")
    g_in.add_argument("--frame1", type=str, help="First image (BGR/RGB)")
    ap.add_argument("--frame2", type=str, help="Second image (required if using --frame1)")

    ap.add_argument("--pairs", type=int, default=1, help="When using --video, number of consecutive pairs to process")
    ap.add_argument("--every", type=int, default=1, help="Frame gap between the pair in --video mode (>=1)")
    ap.add_argument("--start", type=int, default=0, help="Start frame index for video mode")

    ap.add_argument("--block", type=int, default=16, help="Block size for motion estimation")
    ap.add_argument("--search", type=int, default=16, help="Search window (pixels)")

    ap.add_argument("--threshold", type=int, default=None, help="Fixed diff threshold. If omitted, auto-threshold is used.")
    ap.add_argument("--base-threshold", type=int, default=10, help="Base value used by auto-threshold")
    ap.add_argument("--delta-ratio", type=float, default=2.0, help="If %% of pixels over threshold < this value, pick D-frame")

    ap.add_argument("--visualize", action="store_true", help="Show plots")
    args = ap.parse_args()

    # Prepare frames
    pairs_done = 0
    prev_gray = None

    def load_pair(i: int) -> Tuple[np.ndarray, np.ndarray]:
        if args.video:
            return read_two_frames_from_video(args.video, every=args.every, start=args.start + i * args.every)
        else:
            if args.frame2 is None:
                raise ValueError("--frame2 required with --frame1")
            img1 = cv2.imread(args.frame1, cv2.IMREAD_COLOR)
            img2 = cv2.imread(args.frame2, cv2.IMREAD_COLOR)
            if img1 is None or img2 is None:
                raise FileNotFoundError("Could not read frame1 or frame2.")
            return img1, img2

    while pairs_done < args.pairs:
        f1c, f2c = load_pair(pairs_done)

        # First frame of the very first pair => I-frame
        if prev_gray is None:
            curr_gray = to_gray(f1c)
            enc, rec, diag = encode_adaptive(None, curr_gray, args)
            print(f"Encoded {enc.ftype}-frame #0 | diag: {diag}")
            if args.visualize:
                visualize(None, f1c, rec, mv=None, diff=None, mask=None, args=args, title_note="(I-frame)")
            prev_gray = rec.copy()

        # Now process the second frame adaptively
        curr_gray = to_gray(f2c)
        diff = cv2.absdiff(curr_gray, prev_gray)
        thr = args.threshold if args.threshold is not None else auto_threshold(prev_gray, curr_gray, base=args.base_threshold)
        _, mask, st = absolute_diff_stats(prev_gray, curr_gray, thr)
        enc, rec, diag = encode_adaptive(prev_gray, curr_gray, args)
        print(f"Encoded {enc.ftype}-frame #{pairs_done*2+1} | thr={thr} | stats: mean_abs={st.mean_abs:.2f}, over_thr={st.perc_over_thr:.2f}% | diag: {diag}")

        mv = None
        if enc.ftype == "P":
            mv = enc.payload["mv"]

        if args.visualize:
            visualize(prev_gray, f2c, rec, mv=mv, diff=diff, mask=mask, args=args, title_note=f"({enc.ftype}-frame)")

        prev_gray = rec.copy()
        pairs_done += 1


if __name__ == "__main__":
    main()
