"""
mapper.py  -  On-loop aerial mosaic stitcher (single-file build)

Jetson / ROS 2 compatible: NumPy 1.26.4 * OpenCV 4.13.0 * PyTorch 2.8

This file is a self-contained merge of the previous `mapper3.py` and
`compositor.py`, plus a 300-second idle-timeout that saves the final
map and exits once no new frames have arrived for that long.

Top-level user knobs (edit here, no CLI flags needed):
"""
import os

# =============================================================================
#  USER-CONFIGURABLE KNOBS  (the only section you normally need to touch)
# =============================================================================

# Show OpenCV visualisation windows?
#   True  -> live "Current Frame" and "Live Map" windows
#   False -> headless / ROS deployment (no display required)
VISUALISE = True

# Path to the directory where the drone deposits incoming frames.
# The watcher polls this directory and picks up new files as they arrive.
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "4thave")

# Resize factor applied to every frame before processing.
# 0.25 is a good default; lower -> faster; 1.0 -> full resolution (slow).
SCALE = 0.25

# Where to write the final (and periodic auto-save) mosaic.
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_map.png")

# Compute device: 'cuda' (Jetson GPU) or 'cpu'.
DEVICE = "cuda"

# Blending mode: 'seam' (deghosting, recommended) | 'multiband' | 'feather' | 'overwrite'
BLEND_MODE = "seam"

# Motion model: 'homography' (8-DOF, recommended) | 'similarity' (3-DOF)
MOTION_MODEL = "homography"

# Seam finder used when BLEND_MODE='seam':
#   'graphcut' -> best quality  |  'dp' -> good & safe  |  'voronoi' -> auto-remaps to 'dp'
SEAM_FINDER = "voronoi"

# Maximum SuperPoint keypoints per frame. Lower -> less GPU memory.
MAX_KPTS = 1024

# Number of recent frames searched for registration matches.
SEARCH_WINDOW = 5

# Apply CLAHE luminance normalisation on the feature-extraction path.
USE_CLAHE = True

# Ignore yaw / heading metadata (debug only - leave False for real flights).
NO_DEROTATE = False

# -- Live-watcher settings --------------------------------------------------

# Seconds to wait between directory polls when no new frame is available.
POLL_INTERVAL_S = 0.5

# After this many seconds with no new frame the watcher prints a heartbeat.
IDLE_HEARTBEAT_S = 10.0

# After this many seconds with no new frame the watcher saves the final map
# and exits.  Overridable at runtime with IDLE_TIMEOUT_S env var (useful for
# quick tests); defaults to 300 s per deployment spec.
IDLE_TIMEOUT_S = float(os.environ.get("IDLE_TIMEOUT_S", 300.0))

# Auto-save the mosaic every N successfully stitched frames (0 = disable).
AUTOSAVE_EVERY_N = 30

# =============================================================================
#  END OF USER KNOBS  -  do not edit below unless you know what you are doing
# =============================================================================

import math
import copy
import time
import sys
import glob
import csv

# UTF-8 console output (harmless no-op on Linux / Jetson)
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import cv2
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint, utils

# -- NumPy / OpenCV / Torch version guard -------------------------------------
# Emit a clear warning if the environment differs from what we validated on.
def _version_warn(pkg, actual, expected):
    if actual != expected:
        print(f"[WARNING] {pkg} version mismatch: expected {expected}, got {actual}. "
              f"If you see unexpected errors, check your environment.")

_version_warn("numpy",   np.__version__,   "1.26.4")
_version_warn("cv2",     cv2.__version__,   "4.13.0")
_version_warn("torch",   torch.__version__, "2.8.0")

# -----------------------------------------------------------------------------
#  Pipeline tuning constants  (not normally user-facing)
# -----------------------------------------------------------------------------
MIN_RAW_MATCHES     = 15
MIN_INLIERS         = 15
MIN_INLIER_RATIO    = 0.15
MAX_HISTORY         = 15
BORDER_CROP         = 0.04
CANVAS_SIZE         = 12000
CANVAS_HALF         = CANVAS_SIZE // 2
MAX_FOOTPRINT_RATIO = 3.0

FEATHER_BAND_FRAC   = 0.10
SEAM_FEATHER_PX     = 8
GAIN_CLIP           = (0.70, 1.40)
GAIN_DAMPING        = 0.8
GAIN_MIN_OVERLAP    = 500
ROI_PAD             = 4

MIN_KEYPOINTS       = 200
MAX_SKIP_RUN        = 5
REANCHOR_LOOKBACK   = 10


# =============================================================================
#  COMPOSITOR  (inlined from former compositor.py)
# =============================================================================
_SEAM_FINDERS = {
    'graphcut': lambda: cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD'),
    'dp':       lambda: cv2.detail_DpSeamFinder('COLOR_GRAD'),
    'voronoi':  lambda: cv2.detail_VoronoiSeamFinder(),
}

SEAM_TARGET_PX      = 0.10e6
_LUMA_DEADBAND      = 3.0
_AB_DEADBAND        = 2.0
_MIN_OVERLAP_PX     = 800
_BORDER_ERODE_PX    = 4
_DEFAULT_SEAM       = 'dp'


def _to_np(m):
    return m.get() if hasattr(m, 'get') else m


def _auto_scale(h, w):
    return float(min(1.0, max(0.12, (SEAM_TARGET_PX / max(1, h * w)) ** 0.5)))


def _bgr_to_lab_f32(img_f32):
    norm = np.clip(img_f32, 0.0, 255.0) / 255.0
    return cv2.cvtColor(norm.astype(np.float32), cv2.COLOR_BGR2Lab)


def _lab_to_bgr_f32(lab_f32):
    bgr01 = cv2.cvtColor(lab_f32.astype(np.float32), cv2.COLOR_Lab2BGR)
    return np.clip(bgr01 * 255.0, 0.0, 255.0)


def _photometric_match(canvas_f32, warped_f32, overlap_mask, match_color=True):
    if overlap_mask.sum() < _MIN_OVERLAP_PX:
        return warped_f32

    c_lab = _bgr_to_lab_f32(canvas_f32)
    w_lab = _bgr_to_lab_f32(warped_f32)

    c_L = c_lab[:, :, 0][overlap_mask]
    w_L = w_lab[:, :, 0][overlap_mask]
    c_mean, w_mean = float(c_L.mean()), float(w_L.mean())
    c_std = float(c_L.std()) + 1e-5
    w_std = float(w_L.std()) + 1e-5

    out = w_lab
    changed = False

    if abs(c_mean - w_mean) >= _LUMA_DEADBAND:
        alpha = float(np.clip(c_std / w_std, 0.75, 1.35))
        beta = c_mean - alpha * w_mean
        out = out.copy()
        out[:, :, 0] = np.clip(out[:, :, 0] * alpha + beta, 0.0, 100.0)
        changed = True

    if match_color:
        for ch in (1, 2):
            c_m = float(c_lab[:, :, ch][overlap_mask].mean())
            w_m = float(w_lab[:, :, ch][overlap_mask].mean())
            d = c_m - w_m
            if abs(d) >= _AB_DEADBAND:
                if not changed:
                    out = out.copy(); changed = True
                d = float(np.clip(d, -12.0, 12.0))
                out[:, :, ch] = np.clip(out[:, :, ch] + d, -127.0, 127.0)

    if not changed:
        return warped_f32
    return _lab_to_bgr_f32(out)


def _gradient_cost(img_f32):
    L = _bgr_to_lab_f32(img_f32)[:, :, 0]
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    return (mag / (float(mag.max()) + 1e-5)).astype(np.float32)


def _seam_band_blend(canvas_roi, warped, valid_old, valid_new,
                     take_new, union, overlap, feather_px):
    """Hard-copy ownership; feather ONLY a thin band at the seam line."""
    out = canvas_roi.copy()
    out[take_new] = warped[take_new]

    if feather_px <= 0 or not overlap.any():
        out[~union] = 0.0
        return out

    tn = take_new.astype(np.uint8)
    grad = cv2.morphologyEx(tn, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    seam_line = (grad > 0) & overlap
    if not seam_line.any():
        out[~union] = 0.0
        return out

    r = max(1, int(feather_px))
    band = cv2.dilate(seam_line.astype(np.uint8),
                      np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
    band &= overlap

    sigma = max(1.0, feather_px / 2.0)
    a = np.clip(cv2.GaussianBlur(take_new.astype(np.float32), (0, 0), sigma), 0.0, 1.0)
    a3 = a[..., np.newaxis]
    blended = warped * a3 + canvas_roi * (1.0 - a3)
    out[band] = blended[band]

    out[~union] = 0.0
    return out


def seam_blend(canvas_roi, warped, valid_old, valid_new,
               feather_px=16, seam=_DEFAULT_SEAM, seam_scale=None,
               num_bands=None, do_luma_eq=True, match_color=True):
    """Composite `warped` onto `canvas_roi`. float32 BGR [0,255]; masks bool HxW."""
    h, w = canvas_roi.shape[:2]
    union = valid_old | valid_new
    overlap = valid_old & valid_new

    if not overlap.any():
        out = canvas_roi.copy()
        out[valid_new] = warped[valid_new]
        out[~union] = 0.0
        return out.astype(np.float32), union

    if do_luma_eq:
        warped = _photometric_match(canvas_roi, warped, overlap, match_color)

    if seam == 'voronoi':
        print("[compositor] 'voronoi' is unsafe on cv2 4.13.0 -> using 'dp'.")
        seam = 'dp'
    new_wins = np.zeros((h, w), dtype=bool)
    try:
        fx = fy = float(seam_scale) if seam_scale else _auto_scale(h, w)
        small_old = cv2.resize(canvas_roi, None, fx=fx, fy=fy,
                               interpolation=cv2.INTER_AREA).astype(np.float32)
        small_new = cv2.resize(warped, None, fx=fx, fy=fy,
                               interpolation=cv2.INTER_AREA).astype(np.float32)
        sw, sh = small_old.shape[1], small_old.shape[0]
        so = cv2.resize(valid_old.astype(np.uint8) * 255, (sw, sh),
                        interpolation=cv2.INTER_NEAREST)
        sn = cv2.resize(valid_new.astype(np.uint8) * 255, (sw, sh),
                        interpolation=cv2.INTER_NEAREST)

        grad_weight = 0.20
        co = _gradient_cost(small_old)[..., None]
        cn = _gradient_cost(small_new)[..., None]
        sob = np.clip(small_old * (1.0 + grad_weight * co), 0, 255).astype(np.float32)
        snb = np.clip(small_new * (1.0 + grad_weight * cn), 0, 255).astype(np.float32)

        finder = _SEAM_FINDERS.get(seam, _SEAM_FINDERS[_DEFAULT_SEAM])()
        res = finder.find([sob, snb], [(0, 0), (0, 0)], [so, sn])
        sm_new = cv2.resize(_to_np(res[1]), (w, h),
                            interpolation=cv2.INTER_NEAREST)
        new_wins = sm_new > 0
    except cv2.error as exc:
        print("[compositor] seam-finder fallback: {}".format(exc))
        d_new = cv2.distanceTransform(valid_new.astype(np.uint8), cv2.DIST_L2, 3)
        d_old = cv2.distanceTransform(valid_old.astype(np.uint8), cv2.DIST_L2, 3)
        new_wins = d_new >= d_old

    take_new = (valid_new & ~valid_old) | (overlap & new_wins)

    out = _seam_band_blend(canvas_roi, warped, valid_old, valid_new,
                           take_new, union, overlap, feather_px)

    return out.astype(np.float32), union


def exposure_correct_then_blend(canvas_roi, warped, valid_old, valid_new,
                                feather_px=16, seam=_DEFAULT_SEAM,
                                patch_size=64, gain_clip=(0.80, 1.25),
                                do_luma_eq=True):
    """Per-patch L gain before seam_blend. Offline use. ~2-3x slower."""
    h, w = canvas_roi.shape[:2]
    overlap = valid_old & valid_new
    if overlap.sum() < _MIN_OVERLAP_PX:
        return seam_blend(canvas_roi, warped, valid_old, valid_new,
                          feather_px=feather_px, seam=seam, do_luma_eq=do_luma_eq)

    k = np.ones((_BORDER_ERODE_PX * 2 + 1, _BORDER_ERODE_PX * 2 + 1), np.uint8)
    ov_er = cv2.erode(overlap.astype(np.uint8), k).astype(bool)

    gain = np.ones((h, w), np.float32)
    c_lab = _bgr_to_lab_f32(canvas_roi)
    w_lab = _bgr_to_lab_f32(warped)
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            pm = ov_er[y:y + patch_size, x:x + patch_size]
            if pm.sum() < 50:
                continue
            c_L = c_lab[y:y + patch_size, x:x + patch_size, 0][pm]
            w_L = w_lab[y:y + patch_size, x:x + patch_size, 0][pm]
            wm = float(w_L.mean())
            if wm < 1.0:
                continue
            gain[y:y + patch_size, x:x + patch_size] = float(
                np.clip(c_L.mean() / (wm + 1e-4), gain_clip[0], gain_clip[1]))
    gain = cv2.GaussianBlur(gain, (0, 0), patch_size * 0.8)

    w_lab_adj = w_lab.copy()
    w_lab_adj[:, :, 0] = np.clip(w_lab[:, :, 0] * gain, 0.0, 100.0)
    warped_corr = _lab_to_bgr_f32(w_lab_adj)
    return seam_blend(canvas_roi, warped_corr, valid_old, valid_new,
                      feather_px=feather_px, seam=seam, do_luma_eq=False)


# =============================================================================
#  EXIF / metadata helpers
# =============================================================================
def get_gps_info(image_path):
    try:
        img  = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None, None
        geo_info = {}
        for tag, val in exif.items():
            if TAGS.get(tag) == 'GPSInfo':
                for t, v in val.items():
                    geo_info[GPSTAGS.get(t, t)] = v
        if 'GPSLatitude' not in geo_info or 'GPSLongitude' not in geo_info:
            return None, None
        def to_dec(value, ref):
            d, m, s = value
            dec = float(d) + float(m) / 60.0 + float(s) / 3600.0
            return -dec if ref in ('S', 'W') else dec
        lat = to_dec(geo_info['GPSLatitude'],  geo_info.get('GPSLatitudeRef',  'N'))
        lon = to_dec(geo_info['GPSLongitude'], geo_info.get('GPSLongitudeRef', 'E'))
        return lat, lon
    except Exception as e:
        print(f"[WARNING] GPS read failed for {image_path}: {e}")
        return None, None


def get_exif_yaw(image_path):
    """Camera heading from EXIF GPSImgDirection (degrees, 0 = north)."""
    try:
        exif = Image.open(image_path)._getexif()
        if not exif:
            return None
        for tag, val in exif.items():
            if TAGS.get(tag) == 'GPSInfo':
                gps = {GPSTAGS.get(t, t): v for t, v in val.items()}
                d = gps.get('GPSImgDirection')
                if d is not None:
                    return float(d)
        return None
    except Exception:
        return None


def build_yaw_table(image_dir):
    """
    Map image basename -> yaw_deg from a poses.csv sidecar.
    Looks in image_dir and its parent. Returns {} if not found.
    """
    candidates = [
        os.path.join(image_dir, 'poses.csv'),
        os.path.join(os.path.dirname(os.path.normpath(image_dir)), 'poses.csv'),
    ]
    table = {}
    for pc in candidates:
        if not os.path.exists(pc):
            continue
        try:
            with open(pc, newline='') as fh:
                for row in csv.DictReader(fh):
                    fn  = row.get('filename') or row.get('file') or row.get('name')
                    yaw = row.get('yaw_deg')  or row.get('yaw')  or row.get('heading_deg')
                    if fn and yaw not in (None, ''):
                        table[os.path.basename(fn)] = float(yaw)
            if table:
                print(f"[INFO] Loaded {len(table)} yaw entries from {pc}")
                return table
        except Exception as e:
            print(f"[WARNING] Failed to read {pc}: {e}")
    return table


def rotate_to_heading(img, yaw_deg):
    """
    Rotate frame by +yaw_deg so all frames share one canonical heading.
    Lossless 90-deg multiples use cv2.rotate; arbitrary angles use warpAffine.
    """
    a = yaw_deg % 360.0
    if a < 1e-3:
        return img
    if abs(a - 90.0) < 1e-3:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if abs(a - 180.0) < 1e-3:
        return cv2.rotate(img, cv2.ROTATE_180)
    if abs(a - 270.0) < 1e-3:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, M, (nw, nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


# =============================================================================
#  Illumination compensation
# =============================================================================
class IlluminationCompensation:
    """
    Luminance-based ICM. Uses grayscale mean/std so colour-cast terrain
    (golden autumn grass, red-earth roads) doesn't skew the reference.
    """
    def __init__(self):
        self.history_means = []
        self.history_stds  = []
        self.weights       = []

    def _lum_stats(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        m, s = cv2.meanStdDev(gray)
        return float(m[0][0]), float(s[0][0])

    def update_history(self, image, weight):
        m, s = self._lum_stats(image)
        self.history_means.append(m)
        self.history_stds.append(s)
        self.weights.append(float(weight))
        if len(self.history_means) > 5:
            self.history_means.pop(0)
            self.history_stds.pop(0)
            self.weights.pop(0)

    def compensate(self, image):
        if not self.history_means:
            return image
        tw       = sum(self.weights) + 1e-5
        ref_mean = sum(m * w for m, w in zip(self.history_means, self.weights)) / tw
        ref_std  = sum(s * w for s, w in zip(self.history_stds,  self.weights)) / tw
        cur_m, cur_s = self._lum_stats(image)
        cur_s = max(cur_s, 1e-5)
        alpha = float(np.clip(ref_std / cur_s, 0.6, 1.8))
        beta  = ref_mean - alpha * cur_m
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


# =============================================================================
#  Transform helpers
# =============================================================================
def _as3x3(M):
    """Promote a 2x3 affine to 3x3; pass a 3x3 through unchanged."""
    M = np.asarray(M, dtype=np.float64)
    if M.shape == (3, 3):
        return M
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H

_to_3x3 = _as3x3  # backwards-compatible alias

def _compose(A_world, A_rel):
    return _as3x3(A_world) @ _as3x3(A_rel)

def _strip_scale(H_affine):
    theta = math.atan2(H_affine[1, 0], H_affine[0, 0])
    c, s  = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, H_affine[0, 2]],
                     [s,  c, H_affine[1, 2]]], dtype=np.float64)

def _warp_corners(H3, w, h):
    c = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H3.astype(np.float64)).reshape(-1, 2)

def _homography_ok(H3, w, h):
    if H3 is None or not np.isfinite(H3).all():
        return False
    wc   = _warp_corners(H3, w, h).astype(np.float32)
    area = abs(cv2.contourArea(wc))
    if area < 0.2 * w * h or area > 5.0 * w * h:
        return False
    return bool(cv2.isContourConvex(wc))


# =============================================================================
#  Multiband (Laplacian pyramid) blending
# =============================================================================
def _laplacian_blend(img_new, img_old, mask, levels):
    gp_n, gp_o, gp_m = [img_new], [img_old], [mask]
    for _ in range(levels):
        if min(gp_n[-1].shape[:2]) < 8:
            break
        gp_n.append(cv2.pyrDown(gp_n[-1]))
        gp_o.append(cv2.pyrDown(gp_o[-1]))
        gp_m.append(cv2.pyrDown(gp_m[-1]))

    top = len(gp_n) - 1
    out = None
    for i in range(top, -1, -1):
        if i == top:
            ln, lo = gp_n[i], gp_o[i]
        else:
            size = (gp_n[i].shape[1], gp_n[i].shape[0])
            ln = gp_n[i] - cv2.pyrUp(gp_n[i + 1], dstsize=size)
            lo = gp_o[i] - cv2.pyrUp(gp_o[i + 1], dstsize=size)
        m = gp_m[i]
        if m.ndim == 2:
            m = m[..., None]
        band = ln * m + lo * (1.0 - m)
        if out is None:
            out = band
        else:
            out = cv2.pyrUp(out, dstsize=(band.shape[1], band.shape[0])) + band
    return out


# =============================================================================
#  Frame quality scoring
# =============================================================================
def _frame_quality(frame, extractor, device, clahe):
    """
    Score a frame for anchor candidacy.  Returns (score, feats).
    score=0.0 means disqualified (too few keypoints / textureless).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lap_var         = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    brightness_score = 1.0 - abs(mean_brightness - 127.5) / 127.5

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    rgb    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = utils.numpy_image_to_torch(
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    ).to(device)

    with torch.no_grad():
        feats = extractor.extract(tensor)

    n_kpts = int(feats['keypoints'].shape[1])
    if n_kpts < MIN_KEYPOINTS:
        return 0.0, feats

    score = lap_var * brightness_score * (n_kpts / 2048.0)
    return score, feats


# =============================================================================
#  Main pipeline class
# =============================================================================
class DeepGeoROSPipeline:
    def __init__(self, device_str, blend_mode='seam', model='homography',
                 seam='graphcut', use_clahe=True, max_kpts=2048,
                 search_window=SEARCH_WINDOW):

        use_cuda   = device_str == 'cuda' and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        if device_str == 'cuda' and not use_cuda:
            print("[WARNING] CUDA not available, falling back to CPU.")
        print(f"[INFO] SuperPoint/LightGlue on {self.device.type.upper()}")
        print(f"[INFO] Blend={blend_mode}  Model={model}  Seam={seam}  CLAHE={use_clahe}")

        self.extractor    = SuperPoint(max_num_keypoints=max_kpts).eval().to(self.device)
        self.matcher      = LightGlue(features='superpoint').eval().to(self.device)
        self.icm          = IlluminationCompensation()
        self.blend_mode   = blend_mode
        self.model        = model
        self.seam         = seam
        self.use_clahe    = use_clahe
        self.clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.search_window = search_window
        self._frame_hw    = (0, 0)
        self._last_rmse   = float('nan')
        self.last_stats   = {}

        self.global_canvas    = None
        self.weight_canvas    = None
        self.global_transform = np.eye(3, dtype=np.float64)
        self.history          = []

        self.canvas_offset = np.array([[1, 0, CANVAS_HALF],
                                       [0, 1, CANVAS_HALF],
                                       [0, 0, 1]], dtype=np.float64)
        self._weight_cache    = {}
        self._content_bounds  = None

        # Robust anchor state
        self._anchored    = False
        self._skip_run    = 0
        self._skip_buffer = []   # each entry: (comp_frame, feats, score)

    # -- feature extraction ---------------------------------------------------
    def _extract(self, image):
        if self.use_clahe:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = utils.numpy_image_to_torch(rgb).to(self.device)
        return self.extractor.extract(tensor)

    # -- matching -------------------------------------------------------------
    def _match(self, feats0_batched, feats1_batched):
        f0 = copy.copy(feats0_batched)
        f1 = copy.copy(feats1_batched)
        matches_raw         = self.matcher({"image0": f0, "image1": f1})
        f0, f1, matches_raw = [utils.rbd(x) for x in [f0, f1, matches_raw]]
        matches = matches_raw['matches']
        if matches.shape[0] == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        pts0 = f0['keypoints'][matches[:, 0]].cpu().numpy()
        pts1 = f1['keypoints'][matches[:, 1]].cpu().numpy()
        return pts0, pts1

    # -- registration helpers -------------------------------------------------
    @staticmethod
    def _reproj_rmse(H3, pts_curr, pts_past, mask):
        m = mask.ravel().astype(bool)
        if m.sum() == 0:
            return float('nan')
        proj = cv2.perspectiveTransform(
            pts_curr[m].reshape(-1, 1, 2).astype(np.float64), H3
        ).reshape(-1, 2)
        d = proj - pts_past[m]
        return float(np.sqrt((d * d).sum(axis=1).mean()))

    def _estimate(self, pts_curr, pts_past, n_raw):
        w, h = self._frame_hw
        if self.model == 'homography':
            H, mask = cv2.findHomography(pts_curr, pts_past, cv2.USAC_MAGSAC,
                                         3.0, maxIters=2000, confidence=0.999)
            if H is not None and mask is not None:
                inl = int(mask.sum())
                if (inl >= MIN_INLIERS and inl / n_raw >= MIN_INLIER_RATIO
                        and _homography_ok(H, w, h)):
                    return inl, H, self._reproj_rmse(H, pts_curr, pts_past, mask)

        H, mask = cv2.estimateAffinePartial2D(
            pts_curr, pts_past, method=cv2.RANSAC,
            ransacReprojThreshold=4.0, maxIters=3000, confidence=0.99)
        if H is None or mask is None:
            return None
        inl = int(mask.sum())
        if inl < MIN_INLIERS or inl / n_raw < MIN_INLIER_RATIO:
            return None
        H3 = _as3x3(_strip_scale(H))
        return inl, H3, self._reproj_rmse(H3, pts_curr, pts_past, mask)

    def _search(self, feats):
        best_inliers  = -1
        best_H_global = None
        best_rmse     = float('nan')
        for past in list(reversed(self.history))[:self.search_window]:
            pts_past, pts_curr = self._match(past['feats'], feats)
            n_raw = len(pts_past)
            if n_raw < MIN_RAW_MATCHES:
                continue
            est = self._estimate(pts_curr.astype(np.float32),
                                 pts_past.astype(np.float32), n_raw)
            if est is None:
                continue
            inliers, H_rel, rmse = est
            if inliers > best_inliers:
                best_inliers  = inliers
                best_H_global = _compose(past['global_A'], H_rel)
                best_rmse     = rmse
        self._last_rmse = best_rmse
        return best_inliers, best_H_global

    # -- anchor helpers -------------------------------------------------------
    def _commit_anchor(self, comp, feats, label="First anchor"):
        A0 = np.eye(3, dtype=np.float64)
        if self.global_canvas is None:
            self.global_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8)
            self.weight_canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), np.float32)
            A_anchor = A0
        else:
            A_anchor = self.history[-1]['global_A'] if self.history else A0

        self.history.append({'feats': feats, 'global_A': A_anchor})
        if len(self.history) > MAX_HISTORY:
            self.history.pop(0)

        self._stitch(comp, A_anchor)
        self.icm.update_history(comp, 100)
        self._anchored    = True
        self._skip_run    = 0
        self._skip_buffer = []
        print(f"  [ANCHOR] {label} committed.")

    def _flush_skip_buffer(self):
        if not self._skip_buffer:
            return
        print(f"  [FLUSH] Retrying {len(self._skip_buffer)} buffered frame(s)...")
        buf = list(self._skip_buffer)
        self._skip_buffer = []
        for bf, bfeats, bscore in buf:
            bh, bw = bf.shape[:2]
            self._frame_hw = (bw, bh)
            inliers, A_global = self._search(bfeats)
            if A_global is not None:
                wc = _warp_corners(A_global, bw, bh)
                if (not np.isfinite(wc).all()
                        or np.ptp(wc[:, 0]) > MAX_FOOTPRINT_RATIO * bw
                        or np.ptp(wc[:, 1]) > MAX_FOOTPRINT_RATIO * bh):
                    print("    [FLUSH--] Degenerate pose, dropped.")
                    continue
                self.history.append({'feats': bfeats, 'global_A': A_global})
                if len(self.history) > MAX_HISTORY:
                    self.history.pop(0)
                self.icm.update_history(bf, inliers)
                self._stitch(bf, A_global)
                print(f"    [FLUSH-OK] Recovered buffered frame ({inliers} inliers)")
            else:
                print("    [FLUSH--] Still unplaceable, dropped.")

    def _force_reanchor(self):
        if not self._skip_buffer:
            return
        best = max(self._skip_buffer, key=lambda x: x[2])
        bf, bfeats, bscore = best
        if bscore == 0.0:
            print("  [RE-ANCHOR] All buffered frames are textureless - waiting.")
            self._skip_run = 0
            return
        print(f"  [RE-ANCHOR] Bridging gap (score={bscore:.1f}). "
              f"A seam/gap may be visible here.")
        self._commit_anchor(bf, bfeats, label="Re-anchor (gap bridge)")

    # -- per-frame processing -------------------------------------------------
    def process_frame(self, frame, scale, yaw=None):
        if scale != 1.0:
            h, w  = frame.shape[:2]
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        if yaw is not None:
            frame = rotate_to_heading(frame, yaw)

        h, w = frame.shape[:2]
        self._frame_hw = (w, h)
        comp  = self.icm.compensate(frame)
        score, feats = _frame_quality(comp, self.extractor, self.device, self.clahe)

        # -- anchor selection -------------------------------------------------
        if not self._anchored:
            if score == 0.0:
                print("  [PRE-ANCHOR] Poor quality (score=0); buffering...")
                self._skip_buffer.append((comp, feats, score))
                if len(self._skip_buffer) > REANCHOR_LOOKBACK:
                    self._skip_buffer.pop(0)
                return np.zeros((100, 100, 3), np.uint8), comp
            self._commit_anchor(comp, feats, label="First good frame")
            self._flush_skip_buffer()
            return self._crop(), comp

        # -- registration -----------------------------------------------------
        best_inliers, best_A_global = self._search(feats)

        if yaw is None and best_A_global is None:
            comp_180  = cv2.rotate(comp, cv2.ROTATE_180)
            score_180, feats_180 = _frame_quality(
                comp_180, self.extractor, self.device, self.clahe)
            inl_180, A_180 = self._search(feats_180)
            if A_180 is not None and inl_180 > best_inliers:
                comp, feats                 = comp_180, feats_180
                score                       = score_180
                best_inliers, best_A_global = inl_180, A_180

        if best_A_global is not None:
            wc = _warp_corners(best_A_global, w, h)
            if (not np.isfinite(wc).all()
                    or np.ptp(wc[:, 0]) > MAX_FOOTPRINT_RATIO * w
                    or np.ptp(wc[:, 1]) > MAX_FOOTPRINT_RATIO * h):
                print("  [--] Rejected (degenerate accumulated pose)")
                best_A_global = None

        # -- commit or skip ---------------------------------------------------
        if best_A_global is not None:
            self.global_transform = best_A_global
            self.history.append({'feats': feats, 'global_A': best_A_global})
            if len(self.history) > MAX_HISTORY:
                self.history.pop(0)
            self.icm.update_history(comp, best_inliers)
            self._stitch(comp, best_A_global)
            self._skip_run = 0

            cw, ch = self._frame_hw
            ctr = _warp_corners(best_A_global, cw, ch).mean(axis=0)
            self.last_stats = {
                'registered': 1, 'inliers': int(best_inliers),
                'reproj_rmse': self._last_rmse,
                'cx': float(ctr[0]), 'cy': float(ctr[1]),
            }
            print(f"  [OK] Stitched ({best_inliers} inliers, "
                  f"rmse={self._last_rmse:.2f}px)")
            self._flush_skip_buffer()
        else:
            self._skip_run += 1
            self._skip_buffer.append((comp, feats, score))
            if len(self._skip_buffer) > REANCHOR_LOOKBACK:
                self._skip_buffer.pop(0)
            self.last_stats = {
                'registered': 0, 'inliers': 0,
                'reproj_rmse': float('nan'),
                'cx': float('nan'), 'cy': float('nan'),
            }
            print(f"  [--] Skipped (no match, run={self._skip_run})")
            if self._skip_run >= MAX_SKIP_RUN:
                self._force_reanchor()

        return self._crop(), comp

    # -- blending helpers -----------------------------------------------------
    def _frame_weight(self, h, w):
        key = (h, w)
        if key not in self._weight_cache:
            by, bx = int(h * BORDER_CROP), int(w * BORDER_CROP)
            m = np.zeros((h, w), np.uint8)
            m[by:h - by, bx:w - bx] = 255
            dist = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
            self._weight_cache[key] = (m, dist)
        return self._weight_cache[key]

    def _roi_for(self, A_canvas, h, w):
        warped_c = _warp_corners(A_canvas, w, h)
        x0 = int(np.floor(warped_c[:, 0].min())) - ROI_PAD
        y0 = int(np.floor(warped_c[:, 1].min())) - ROI_PAD
        x1 = int(np.ceil(warped_c[:, 0].max()))  + ROI_PAD
        y1 = int(np.ceil(warped_c[:, 1].max()))  + ROI_PAD
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(CANVAS_SIZE, x1); y1 = min(CANVAS_SIZE, y1)
        return x0, y0, x1, y1

    def _stitch(self, frame, A_global):
        h, w = frame.shape[:2]
        _crop_mask, weight = self._frame_weight(h, w)

        A_canvas = _compose(self.canvas_offset, A_global)
        x0, y0, x1, y1 = self._roi_for(A_canvas, h, w)
        if x1 <= x0 or y1 <= y0:
            print("  [WARNING] frame fell outside canvas, skipped paint")
            return

        T     = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
        A_roi = T @ A_canvas
        bw, bh = x1 - x0, y1 - y0

        warped = cv2.warpPerspective(frame, A_roi, (bw, bh),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0)).astype(np.float32)
        w_new  = cv2.warpPerspective(weight, A_roi, (bw, bh),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        valid_new = w_new > 1e-3

        canvas_roi = self.global_canvas[y0:y1, x0:x1].astype(np.float32)
        w_old      = self.weight_canvas[y0:y1, x0:x1]
        valid_old  = w_old > 1e-3

        gain_zone = valid_new & (w_old > 2.0) & (w_new > 2.0)
        if gain_zone.sum() >= GAIN_MIN_OVERLAP:
            mean_canvas = canvas_roi[gain_zone].reshape(-1, 3).mean(axis=0)
            mean_new    = warped[gain_zone].reshape(-1, 3).mean(axis=0)
            gain = (mean_canvas + 1e-3) / (mean_new + 1e-3)
            gain = 1.0 + (gain - 1.0) * GAIN_DAMPING
            gain = np.clip(gain, GAIN_CLIP[0], GAIN_CLIP[1])
            warped = np.clip(warped * gain[None, None, :], 0, 255)

        if self.blend_mode == 'seam':
            out, _ = seam_blend(canvas_roi, warped, valid_old, valid_new,
                                feather_px=SEAM_FEATHER_PX, seam=self.seam)
        elif self.blend_mode == 'overwrite':
            paint = np.any(warped > 0, axis=2)
            canvas_roi[paint] = warped[paint]
            out = canvas_roi
        else:
            band  = FEATHER_BAND_FRAC * min(h, w)
            alpha = np.clip((w_new - w_old) / band * 0.5 + 0.5, 0.0, 1.0)
            alpha[~valid_new] = 0.0
            alpha[valid_new & ~valid_old] = 1.0
            if self.blend_mode == 'feather':
                a3  = alpha[..., None]
                out = warped * a3 + canvas_roi * (1.0 - a3)
            else:   # multiband
                new_f = warped.copy()
                old_f = canvas_roi.copy()
                new_f[~valid_new] = canvas_roi[~valid_new]
                old_f[~valid_old & valid_new] = warped[~valid_old & valid_new]
                levels = max(2, min(6, int(math.log2(max(16, min(bh, bw)))) - 3))
                out = _laplacian_blend(new_f, old_f, alpha, levels)
                out = np.clip(out, 0, 255)

        union = valid_new | valid_old
        self.global_canvas[y0:y1, x0:x1][union] = out[union].astype(np.uint8)
        self.weight_canvas[y0:y1, x0:x1] = np.maximum(w_old, w_new)

        if self._content_bounds is None:
            self._content_bounds = [x0, y0, x1, y1]
        else:
            b = self._content_bounds
            b[0] = min(b[0], x0); b[1] = min(b[1], y0)
            b[2] = max(b[2], x1); b[3] = max(b[3], y1)

    def _crop(self):
        if self._content_bounds is None:
            return self.global_canvas
        x0, y0, x1, y1 = self._content_bounds
        return self.global_canvas[y0:y1, x0:x1]


# =============================================================================
#  Live directory watcher
# =============================================================================
def _glob_images(directory):
    """Return a sorted list of all .jpg / .jpeg / .png files in directory."""
    patterns = ['*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.png', '*.PNG']
    found = []
    for pat in patterns:
        found.extend(glob.glob(os.path.join(directory, pat)))
    return sorted(set(found))


def _show(title, img, max_side=1200):
    """Resize-to-fit then imshow. No-op if VISUALISE is False."""
    if not VISUALISE:
        return
    if max(img.shape[:2]) > max_side:
        s   = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)
    cv2.imshow(title, img)


def run_live(pipeline, image_dir, scale, yaw_table, out_path):
    """
    Poll image_dir for new frames.  Process each exactly once in sorted order.
    When all current images are consumed, wait POLL_INTERVAL_S and try again.
    After IDLE_TIMEOUT_S seconds with no new frame the final mosaic is saved
    and the function returns.  Press 'q' in the visualisation window (or
    Ctrl-C) to stop earlier.
    """
    processed     = set()          # basenames already handled
    mosaic        = None
    n_stitched    = 0              # successfully stitched frame count
    t_start       = time.time()
    t_last_new    = t_start        # last time we actually saw a new frame
    t_last_beat   = t_start        # last time we printed an idle heartbeat

    print(f"[INFO] Watching {image_dir} for incoming frames "
          f"(poll every {POLL_INTERVAL_S}s, idle-timeout {IDLE_TIMEOUT_S:.0f}s)...")
    if VISUALISE:
        print("[INFO] Press 'q' in the display window to stop.")
    else:
        print("[INFO] Headless mode - press Ctrl-C to stop.")

    try:
        while True:
            all_paths = _glob_images(image_dir)
            new_paths = [p for p in all_paths
                         if os.path.basename(p) not in processed]

            if not new_paths:
                now = time.time()
                idle = now - t_last_new

                # Idle-timeout: save + exit
                if idle >= IDLE_TIMEOUT_S:
                    print(f"[INFO] No new frames for {idle:.0f}s "
                          f">= {IDLE_TIMEOUT_S:.0f}s idle-timeout - "
                          f"finalising map.")
                    break

                # Heartbeat every IDLE_HEARTBEAT_S seconds
                if now - t_last_beat > IDLE_HEARTBEAT_S:
                    print(f"[WAIT] No new frames for {idle:.0f}s "
                          f"({len(processed)} processed, "
                          f"{IDLE_TIMEOUT_S - idle:.0f}s until timeout)...")
                    t_last_beat = now

                # Keep the display alive and check for 'q'
                if VISUALISE:
                    if cv2.waitKey(int(POLL_INTERVAL_S * 1000)) == ord('q'):
                        print("[INFO] 'q' pressed - stopping.")
                        break
                else:
                    time.sleep(POLL_INTERVAL_S)
                continue

            # New frames available - process them in order
            t_last_new = time.time()
            for path in new_paths:
                base = os.path.basename(path)

                # Wait until the file is fully written (size stable for 0.1 s)
                try:
                    sz0 = os.path.getsize(path)
                    time.sleep(0.10)
                    sz1 = os.path.getsize(path)
                    if sz0 != sz1:
                        # Still being written - come back next poll
                        continue
                except OSError:
                    continue

                frame = cv2.imread(path)
                processed.add(base)
                # File was fully readable - update last-seen timestamp so
                # the idle-timeout only measures gaps after this frame.
                t_last_new = time.time()

                if frame is None:
                    print(f"  [SKIP] Unreadable: {base}")
                    continue

                yaw = None
                if not NO_DEROTATE:
                    yaw = yaw_table.get(base) or get_exif_yaw(path)

                total = len(processed)
                print(f"[{total}] {base}" +
                      (f"  yaw={yaw:.0f}" if yaw is not None else "  yaw=?"))

                mosaic, cur_frame = pipeline.process_frame(
                    frame, scale=scale, yaw=yaw)

                if pipeline.last_stats.get('registered', 0):
                    n_stitched += 1

                # -- visualisation (gated on VISUALISE) ----------------------
                _show("Current Frame", cur_frame)
                if mosaic is not None and mosaic.size > 0:
                    _show("Live Map", mosaic)

                if VISUALISE:
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        print("[INFO] 'q' pressed - stopping.")
                        if mosaic is not None:
                            cv2.imwrite(out_path, mosaic)
                            print(f"[INFO] Saved {out_path}")
                        cv2.destroyAllWindows()
                        return

                # -- periodic auto-save --------------------------------------
                if (AUTOSAVE_EVERY_N > 0
                        and n_stitched > 0
                        and n_stitched % AUTOSAVE_EVERY_N == 0
                        and mosaic is not None):
                    cv2.imwrite(out_path, mosaic)
                    print(f"[INFO] Auto-saved {out_path} "
                          f"({n_stitched} stitched, "
                          f"{time.time() - t_start:.0f}s elapsed)")

            # Reset heartbeat clock; the next idle window starts now.
            t_last_beat = time.time()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    # -- final save -----------------------------------------------------------
    dt = time.time() - t_start
    print(f"[INFO] Done. {len(processed)} frames seen, "
          f"{n_stitched} stitched, {dt:.1f}s elapsed.")
    if mosaic is not None:
        cv2.imwrite(out_path, mosaic)
        print(f"[INFO] Final mosaic saved -> {out_path}")
    if VISUALISE:
        cv2.destroyAllWindows()


# =============================================================================
#  Entry point
# =============================================================================
def main():
    if not os.path.isdir(IMAGE_DIR):
        print(f"[ERROR] IMAGE_DIR does not exist: {IMAGE_DIR}")
        sys.exit(1)

    yaw_table = {} if NO_DEROTATE else build_yaw_table(IMAGE_DIR)

    pipeline = DeepGeoROSPipeline(
        DEVICE,
        blend_mode=BLEND_MODE,
        model=MOTION_MODEL,
        seam=SEAM_FINDER,
        use_clahe=USE_CLAHE,
        max_kpts=MAX_KPTS,
        search_window=SEARCH_WINDOW,
    )

    run_live(pipeline, IMAGE_DIR, SCALE, yaw_table, OUTPUT_PATH)


if __name__ == "__main__":
    main()
