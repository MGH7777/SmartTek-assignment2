# app1.py
import os, io, zipfile, json, math
import numpy as np
from PIL import Image

try:
    import pydicom
except Exception:
    pydicom = None

# -------------------- Huffman Coding --------------------
class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right
    def __lt__(self, other):
        return self.freq < other.freq

def _build_huffman_tree(freqs):
    import heapq
    heap = []
    for s, f in freqs.items():
        heapq.heappush(heap, HuffmanNode(symbol=s, freq=f))
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, HuffmanNode(freq=a.freq + b.freq, left=a, right=b))
    return heap[0] if heap else None

def _build_codebook(node, prefix="", table=None):
    if table is None: table = {}
    if node is None: return table
    if node.symbol is not None and node.left is None and node.right is None:
        table[node.symbol] = prefix or "0"
        return table
    if node.left: _build_codebook(node.left, prefix + "0", table)
    if node.right: _build_codebook(node.right, prefix + "1", table)
    return table

def huffman_encode(symbols):
    from collections import Counter
    freqs = Counter(symbols)
    root = _build_huffman_tree(freqs)
    codebook = _build_codebook(root)
    bitstream = "".join(codebook[s] for s in symbols)
    byts = int(bitstream, 2).to_bytes((len(bitstream) + 7) // 8, "big") if bitstream else b""
    meta = {"codebook": {str(k): v for k, v in codebook.items()}, "bitlen": len(bitstream)}
    return byts, meta

def huffman_decode(byts, meta):
    bitlen = meta["bitlen"]
    codebook = {int(k): v for k, v in meta["codebook"].items()}
    # Build decoding trie
    trie = {}
    for sym, code in codebook.items():
        node = trie
        for b in code:
            node = node.setdefault(b, {})
        node["$"] = sym
    # Bits to string
    full_bits = bin(int.from_bytes(byts, "big"))[2:]
    pad = (len(byts) * 8) - len(full_bits)
    full_bits = "0" * pad + full_bits
    
    bits = full_bits[-bitlen:] if bitlen > 0 else "" 
    
    out, node = [], trie
    for b in bits:
        node = node.get(b)
        if node is None: break
        if "$" in node:
            out.append(node["$"])
            node = trie
    return out

# -------------------- DCT helpers --------------------
def dct_2d(block):
    """2D DCT implementation using scipy if available, otherwise numpy"""
    try:
        from scipy.fftpack import dct
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    except ImportError:
        # Fallback to numpy implementation
        return _dct_2d_numpy(block)

def idct_2d(block):
    """2D Inverse DCT implementation using scipy if available, otherwise numpy"""
    try:
        from scipy.fftpack import idct
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    except ImportError:
        # Fallback to numpy implementation
        return _idct_2d_numpy(block)

def _dct_2d_numpy(block):
    """Numpy implementation of 2D DCT"""
    M, N = block.shape
    dct_result = np.zeros((M, N))
    
    for u in range(M):
        for v in range(N):
            sum_val = 0.0
            for i in range(M):
                for j in range(N):
                    cos1 = np.cos((2 * i + 1) * u * np.pi / (2 * M))
                    cos2 = np.cos((2 * j + 1) * v * np.pi / (2 * N))
                    sum_val += block[i, j] * cos1 * cos2
            
            cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
            cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
            dct_result[u, v] = 2.0 / np.sqrt(M * N) * cu * cv * sum_val
    
    return dct_result

def _idct_2d_numpy(block):
    """Numpy implementation of 2D Inverse DCT"""
    M, N = block.shape
    idct_result = np.zeros((M, N))
    
    for i in range(M):
        for j in range(N):
            sum_val = 0.0
            for u in range(M):
                for v in range(N):
                    cu = 1.0 / np.sqrt(2) if u == 0 else 1.0
                    cv = 1.0 / np.sqrt(2) if v == 0 else 1.0
                    cos1 = np.cos((2 * i + 1) * u * np.pi / (2 * M))
                    cos2 = np.cos((2 * j + 1) * v * np.pi / (2 * N))
                    sum_val += cu * cv * block[u, v] * cos1 * cos2
            
            idct_result[i, j] = 2.0 / np.sqrt(M * N) * sum_val
    
    return idct_result

def quality_to_scale(quality):
    """Convert quality (1-100) to quantization scale factor"""
    quality = max(1, min(quality, 100))
    if quality <= 50:
        return 50.0 / quality
    else:
        return 2.0 - (quality / 50.0)

# Standard JPEG quantization table
JPEG_QTABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def quantize_dct(dct_coeffs, quality):
    """Quantize DCT coefficients based on quality"""
    scale = quality_to_scale(quality)
    qtable = np.clip(JPEG_QTABLE * scale, 1, 255)
    quantized = np.round(dct_coeffs / qtable).astype(np.int32)
    return quantized, qtable

def dequantize_dct(quantized, qtable):
    """Dequantize DCT coefficients"""
    return quantized * qtable

# Zigzag scan pattern
def zigzag_indices(n=8):
    """Generate zigzag scan indices"""
    indices = []
    for i in range(n * 2):
        if i < n:
            x = 0
            y = i
        else:
            x = i - n + 1
            y = n - 1
        while x < n and y >= 0:
            if i % 2 == 0:
                indices.append((y, x))
            else:
                indices.append((x, y))
            x += 1
            y -= 1
    return indices[:n*n]

ZZ_ORDER = zigzag_indices(8)

def zigzag_scan(block):
    """Convert 8x8 block to zigzag order"""
    return np.array([block[i, j] for i, j in ZZ_ORDER])

def inverse_zigzag(arr):
    """Convert zigzag array back to 8x8 block"""
    block = np.zeros((8, 8), dtype=arr.dtype)
    for idx, (i, j) in enumerate(ZZ_ORDER):
        block[i, j] = arr[idx]
    return block

def rle_encode(arr):
    """Run-length encode an array"""
    if len(arr) == 0:
        return []
    
    encoded = []
    count = 1
    current = arr[0]
    
    for i in range(1, len(arr)):
        if arr[i] == current and count < 255:
            count += 1
        else:
            encoded.extend([current, count])
            current = arr[i]
            count = 1
    
    encoded.extend([current, count])
    return encoded

def rle_decode(encoded):
    """Run-length decode an array"""
    if len(encoded) == 0:
        return []
    
    decoded = []
    for i in range(0, len(encoded), 2):
        if i + 1 < len(encoded):
            value = encoded[i]
            count = encoded[i + 1]
            decoded.extend([value] * count)
    return decoded

# -------------------- DCT Compression --------------------
def compress_dct(img8, quality=50, use_huffman=True, bs=8):
    """Compress image using DCT"""
    # Convert to float and center around 0
    img_float = img8.astype(np.float64) - 128.0
    
    H, W = img8.shape
    # Pad image to be multiple of block size
    H_pad = ((H + bs - 1) // bs) * bs
    W_pad = ((W + bs - 1) // bs) * bs
    
    padded = np.zeros((H_pad, W_pad), dtype=np.float64)
    padded[:H, :W] = img_float
    
    # Get quantization table
    _, qtable = quantize_dct(np.zeros((bs, bs)), quality)
    
    all_blocks = []
    
    # Process each block
    for y in range(0, H_pad, bs):
        for x in range(0, W_pad, bs):
            block = padded[y:y+bs, x:x+bs]
            
            # Apply DCT
            dct_coeffs = dct_2d(block)
            
            # Quantize
            quantized, _ = quantize_dct(dct_coeffs, quality)
            
            # Zigzag scan
            zz = zigzag_scan(quantized)
            all_blocks.append(zz)
    
    # Flatten all zigzag arrays and RLE encode
    flattened = []
    for block in all_blocks:
        flattened.extend(block)
    
    # RLE encode
    rle_encoded = rle_encode(flattened)
    
    # Huffman encode if requested
    if use_huffman:
        compressed_data, huff_meta = huffman_encode(rle_encoded)
    else:
        compressed_data = np.array(rle_encoded, dtype=np.int32).tobytes()
        huff_meta = {"bitlen": len(compressed_data) * 8, "codebook": {}}
    
    # Prepare metadata
    meta = {
        "H": H, "W": W, 
        "H_pad": H_pad, "W_pad": W_pad,
        "bs": bs, "quality": quality, 
        "use_huffman": use_huffman,
        "qtable": qtable.tolist(),
        "huff_meta": huff_meta
    }
    
    return compressed_data, meta

def decompress_dct(compressed_data, meta):
    """Decompress image using DCT"""
    H, W = meta["H"], meta["W"]
    H_pad, W_pad = meta["H_pad"], meta["W_pad"]
    bs = meta["bs"]
    qtable = np.array(meta["qtable"])
    use_huffman = meta["use_huffman"]
    
    # Decode the data
    if use_huffman:
        rle_decoded = huffman_decode(compressed_data, meta["huff_meta"])
    else:
        rle_decoded = np.frombuffer(compressed_data, dtype=np.int32).tolist()
    
    # RLE decode
    flattened = rle_decode(rle_decoded)
    
    # Reconstruct blocks
    num_blocks = (H_pad // bs) * (W_pad // bs)
    blocks = []
    
    for i in range(num_blocks):
        start_idx = i * 64
        end_idx = start_idx + 64
        if end_idx <= len(flattened):
            zz_block = flattened[start_idx:end_idx]
        else:
            # Pad with zeros if incomplete
            zz_block = flattened[start_idx:] + [0] * (64 - len(flattened[start_idx:]))
        
        # Inverse zigzag
        quantized_block = inverse_zigzag(np.array(zz_block, dtype=np.int32))
        
        # Dequantize
        dct_block = dequantize_dct(quantized_block, qtable)
        
        # Inverse DCT
        reconstructed_block = idct_2d(dct_block)
        blocks.append(reconstructed_block)
    
    # Reconstruct image from blocks
    reconstructed = np.zeros((H_pad, W_pad), dtype=np.float64)
    block_idx = 0
    
    for y in range(0, H_pad, bs):
        for x in range(0, W_pad, bs):
            if block_idx < len(blocks):
                reconstructed[y:y+bs, x:x+bs] = blocks[block_idx]
                block_idx += 1
    
    # Remove padding and convert back to uint8
    result = reconstructed[:H, :W] + 128.0
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

# -------------------- FFT Compression --------------------
def radial_quant_matrix(shape, strength=1.0):
    H, W = shape
    cy, cx = H//2, W//2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing='ij')
    R = np.sqrt(X*X + Y*Y) / (np.sqrt(cx*cx + cy*cy) + 1e-8)
    return 1.0 + strength * (R**1.5) * 50.0


def compress_fft(img8, quality=50, use_huffman=True):
    """
    FFT compressor with DCT-like quality behavior PLUS stronger, visible effect:
    - JPEG-like qf from quality_to_scale()
    - Low-pass keep radius grows with Quality (keeps more low-freq at high Q)
    - Dead-zone threshold at lower qualities to zero tiny coeffs
    - Length-prefixed value/count RLE, then optional Huffman (at the end)
    """
    import numpy as np

    img = img8.astype(np.float64)
    H, W = img.shape

    # Centered FFT
    Fshift = np.fft.fftshift(np.fft.fft2(img))

    # JPEG-like scaling (same direction as DCT): higher Q -> smaller qf
    qf = max(0.1, float(quality_to_scale(int(quality))))  # reuse mapping

    # Radial weighting table (acts like a "quant table" in freq domain)
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    Rn = R / (R.max() + 1e-8)
    k, p = 60.0, 1.5
    Qrad = 1.0 + k * (Rn ** p)

    # Final divisor: EXACT analogue of DCT's (qtable * qf)
    denom = Qrad * qf

    # ---- Low-pass keep radius tied to Quality (biggest visual effect) ----
    #  Q=100 -> keep almost everything (~0.95 of radius)
    #  Q=  1 -> keep very little (~0.10 of radius)
    keep_radius = 0.10 + 0.85 * (quality / 100.0)      # 0.10..0.95
    mask_keep = (Rn <= keep_radius)

    # Quantize ONLY within the keep radius; zero the rest outright
    Rq = np.zeros((H, W), np.int32)
    Iq = np.zeros((H, W), np.int32)
    Rq[mask_keep] = np.round(np.real(Fshift)[mask_keep] / denom[mask_keep]).astype(np.int32)
    Iq[mask_keep] = np.round(np.imag(Fshift)[mask_keep] / denom[mask_keep]).astype(np.int32)

    # ---- Dead-zone threshold to push more zeros at lower quality ----
    # (grows as quality decreases; no dead-zone at high Q)
    strength = (100 - max(1, min(quality, 100))) / 100.0  # 0..1
    if strength > 0.2:
        thr = 1 + int(8 * (strength - 0.2) / 0.8)         # 1..9 roughly
        Rq[np.abs(Rq) <= thr] = 0
        Iq[np.abs(Iq) <= thr] = 0

    # ---- Serialize as value,count RLE over flat arrays (length-prefixed R) ----
    def rle_vc_encode(arr1d):
        out = []
        if not arr1d:
            return out
        run_val = arr1d[0]
        run_len = 1
        for v in arr1d[1:]:
            if v == run_val:
                run_len += 1
            else:
                out.extend([int(run_val), int(run_len)])
                run_val = v
                run_len = 1
        out.extend([int(run_val), int(run_len)])
        return out

    symR = rle_vc_encode(Rq.flatten().tolist())
    symI = rle_vc_encode(Iq.flatten().tolist())
    symbols = [len(symR)] + symR + symI

    if use_huffman:
        byts, huff_meta = huffman_encode(symbols)
    else:
        arr = np.array(symbols, dtype=np.int32)
        byts = arr.tobytes()
        huff_meta = {"bitlen": len(byts) * 8, "codebook": {}}

    # Store params needed by the decoder (matches current decompress_fft)
    return byts, {
        "H": int(H), "W": int(W),
        "use_huffman": bool(use_huffman),
        "quality": int(quality),
        "qf": float(qf),
        "k": float(k), "p": float(p),
        "huff_meta": huff_meta,
    }


def decompress_fft(byts, meta):
    """
    Inverse of the compressor above:
    - Decode Huffman (if used)
    - Split length-prefixed R/I streams
    - RLE-decode to FLAT arrays, reshape to (H,W)
    - Dequantize with the SAME denom = (Qrad * qf)
    - IFFT and clamp to 0..255 uint8
    """
    import numpy as np

    H, W = int(meta["H"]), int(meta["W"])

    # Restore symbol stream
    if meta.get("use_huffman", True):
        symbols = huffman_decode(byts, meta["huff_meta"])
    else:
        symbols = np.frombuffer(byts, dtype=np.int32).tolist()
    if not symbols:
        raise ValueError("Empty FFT stream")

    # Split: [len(symR)] + symR + symI
    lenR = int(symbols[0])
    if lenR < 0 or 1 + lenR > len(symbols):
        raise ValueError("Corrupt FFT stream: bad R-length")
    symR = symbols[1:1 + lenR]
    symI = symbols[1 + lenR:]

    # RLE (value,count) decode to FLAT arrays
    def rle_vc_decode(sym):
        out = []
        it = iter(sym)
        for v in it:
            try:
                c = next(it)
            except StopIteration:
                break  # ignore dangling value
            out.extend([int(v)] * int(c))
        return out

    Rq_flat = rle_vc_decode(symR)
    Iq_flat = rle_vc_decode(symI)

    need = H * W
    if len(Rq_flat) < need: Rq_flat += [0] * (need - len(Rq_flat))
    if len(Iq_flat) < need: Iq_flat += [0] * (need - len(Iq_flat))
    Rq = np.array(Rq_flat[:need], dtype=np.int32).reshape(H, W)
    Iq = np.array(Iq_flat[:need], dtype=np.int32).reshape(H, W)

    # --- SAME denom as encoder: denom = Qrad * qf ---
    qf = float(meta["qf"])
    k = float(meta.get("k", 60.0))
    p = float(meta.get("p", 1.5))

    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    Y, X = np.meshgrid(y, x, indexing="ij")
    R = np.sqrt(X * X + Y * Y)
    Rn = R / (R.max() + 1e-8)
    Qrad = 1.0 + k * (Rn ** p)
    denom = Qrad * qf

    # Dequantize & inverse FFT
    Fshift = (Rq * denom).astype(np.float64) + 1j * (Iq * denom).astype(np.float64)
    rec = np.fft.ifft2(np.fft.ifftshift(Fshift)).real
    return np.clip(rec, 0, 255).astype(np.uint8)



# -------------------- Metrics --------------------
def mse(a, b): 
    return np.mean((a.astype(float) - b.astype(float))**2)

def psnr(a, b, maxval=255.0): 
    m = mse(a, b)
    return 20 * np.log10(maxval) - 10 * np.log10(m) if m > 0 else float("inf")

def ssim(a, b):
    a, b = a.astype(float), b.astype(float)
    mu_x, mu_y = a.mean(), b.mean()
    sigma_x, sigma_y = a.var(), b.var()
    sigma_xy = ((a - mu_x) * (b - mu_y)).mean()
    K1, K2, L = 0.01, 0.03, 255
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return float(num / den)

# -------------------- DICOM I/O --------------------
def load_first_dicom_from_zip(zf):
    for name in zf.namelist():
        if name.lower().endswith(".dcm"):
            with zf.open(name) as f:
                ds = pydicom.dcmread(io.BytesIO(f.read()))
                arr = ds.pixel_array
                return arr, getattr(ds, "BitsStored", 16), name
    raise ValueError("No .dcm files found.")

def to_uint8(img):
    img = img.astype(float)
    mn, mx = img.min(), img.max()
    if mx <= mn:
        return np.zeros_like(img, np.uint8)
    return np.clip((255 * (img - mn) / (mx - mn)).round(), 0, 255).astype(np.uint8)