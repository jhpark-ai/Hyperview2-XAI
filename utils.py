from scipy.signal import savgol_filter
from scipy.ndimage import sobel
from skimage.feature import graycomatrix, graycoprops
from glob import glob
from scipy.linalg import svd  # for SVD
from pathlib import Path
from numpy.linalg import svd, LinAlgError
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings("ignore")


def remove_bad_bands_hsi(hsi_vec: np.ndarray) -> np.ndarray:
    """
    Function to remove bad bands from HSI Satellite
    Remove bands 98~109, 130~149
    Return 198 bands

    Parameters
    ----------
    hsi_vec : np.ndarray
        HSI Satellite vector
    """
    if hsi_vec is None:
        return np.zeros(198, dtype=np.float32)
    if hsi_vec.shape[0] != 230:
        if hsi_vec.shape[0] == 198:
            return hsi_vec
        else:
            vec = np.zeros(230, dtype=np.float32)
            vec[: hsi_vec.shape[0]] = hsi_vec[:230]
            hsi_vec = vec
    v = np.concatenate([hsi_vec[:98], hsi_vec[110:130], hsi_vec[150:]])
    # v → 98 + 20 + (230 - 150 - 1 + 1) = 98 + 20 + 80 = 198
    return v.astype(np.float32)


# ==============================
# 3. Dataset preprocessing: feature/label matrix construction
# ==============================
def compute_derivatives(data: np.ndarray, order: int) -> np.ndarray:
    """
    Calculate nth derivative of data.
    Implement by applying np.gradient n times.
    """
    # np.gradient only calculates 1st derivative, so apply n times
    result = data
    for _ in range(order):
        # edge_order is only 1 or 2, so use default value (1)
        result = np.gradient(result, axis=0)
    return result


def resize_patch(data: np.ndarray, mask: np.ndarray, target_h: int, target_w: int) -> tuple:
    """
    Resize data patch to target size
    - If target size is smaller, upsample (interpolation).
    - If target size is larger, downsample (block averaging).

    Parameters
    ----------
    data : np.ndarray
        Data patch
    mask : np.ndarray
    """
    _, H, W = data.shape

    # Downsample (Block Averaging)
    if H > target_h or W > target_w:
        h_edges = np.linspace(0, H, target_h + 1, dtype=int)
        w_edges = np.linspace(0, W, target_w + 1, dtype=int)

        new_data = np.zeros(
            (data.shape[0], target_h, target_w), dtype=data.dtype)
        new_mask = np.zeros(
            (mask.shape[0], target_h, target_w), dtype=mask.dtype)

        for i in range(target_h):
            h0, h1 = h_edges[i], h_edges[i+1]
            for j in range(target_w):
                w0, w1 = w_edges[j], w_edges[j+1]

                # Data processing
                block_data = data[:, h0:h1, w0:w1]
                if block_data.size > 0:
                    new_data[:, i, j] = block_data.mean(axis=(1, 2))

                # Mask processing
                block_mask = mask[:, h0:h1, w0:w1]
                if block_mask.size > 0:
                    new_mask[:, i, j] = block_mask.mean(axis=(1, 2))

        return new_data, new_mask > 0.5

    # Upsample (Interpolation)
    elif H < target_h or W < target_w:
        # scikit-image's resize does not support (C, H, W) directly, so we transpose to (H, W, C)
        data_res = cv2.resize(data.transpose(1, 2, 0),  dsize=(
            target_h, target_w), interpolation=cv2.INTER_NEAREST)
        # resize mask - convert to uint8 to fix boolean type issue
        mask_original_dtype = mask.dtype
        mask_transposed = mask.transpose(1, 2, 0)
        # convert boolean or other types to uint8
        if mask_transposed.dtype == bool:
            mask_uint8 = mask_transposed.astype(np.uint8) * 255
        else:
            mask_uint8 = mask_transposed.astype(np.uint8)

        mask_res = cv2.resize(mask_uint8, dsize=(
            target_h, target_w), interpolation=cv2.INTER_NEAREST)

        # restore original type
        if mask_original_dtype == bool:
            mask_res = (mask_res > 127).astype(mask_original_dtype)
        else:
            mask_res = mask_res.astype(mask_original_dtype)

        return data_res.astype(data.dtype).transpose(2, 0, 1), mask_res.transpose(2, 0, 1)
    # If size is the same
    else:
        return data, mask


def fft_low_k(x: np.ndarray, k: int, axis: int = -1,
              return_amp_phase: bool = False):
    """
    Extract low-frequency components (including DC) from the front

    Parameters
    ----------
    x : np.ndarray
        Real input array. 1-D or multi-dimensional.
    k : int
        Number of frequency components to keep (including 0 Hz). 0 < k ≤ N//2 + 1.
    axis : int, optional
        Axis to apply FFT. Default is the last axis (-1).
    return_amp_phase : bool, optional
        • False (default) → return k complex FFT coefficients
        • True  → return (amplitude, phase) tuple

    Returns
    -------
    np.ndarray | tuple(np.ndarray, np.ndarray)
        - `return_amp_phase=False` → return k complex FFT coefficients
          (shape: same as input but axis length is k)
        - `True` → (amp, phase) : each shape = (..., k)

    Notes
    -----
    * `np.fft.rfft` is used to calculate only the **positive frequency range** (0 ~ Nyquist)
      so all information of the real input is preserved and the length is `N//2+1` which is half of the original length.
    * If k is out of range, `ValueError` is raised.
    """
    x = np.asarray(x)
    spec = np.fft.rfft(x, axis=axis)        # positive frequency spectrum
    n_freq = spec.shape[axis]

    if k < 1 or k > n_freq:
        raise ValueError(f"k must be in the range [1, {n_freq}] (got {k}).")

    # move axis to front for slicing → restore after slicing
    spec = np.moveaxis(spec, axis, -1)[..., :k]
    spec = np.moveaxis(spec, -1, axis)

    if return_amp_phase:
        amp = np.abs(spec)
        phase = np.angle(spec)
        return amp, phase
    else:
        return spec


def mn_specific_features(msi_mean: np.ndarray) -> np.ndarray:
    """
    Extract Mn-specific features - enhance Mn-specific spectral characteristics

    Parameters
    ----------
    hsi_mean : (198,) PRISMA spectrum (400-2500 nm, 10 nm step) after bad band removal
    msi_mean : (13,) Sentinel-2 13 bands (Upsampled to 10 m)

    Returns
    -------
    np.ndarray - expanded Mn-specific features
    """

    features = []

    # existing features
    B2, B3 = 1, 2  # 490, 560 nm
    B2_B3_ratio = msi_mean[B2] / (msi_mean[B3] + 1e-9)
    features.append(B2_B3_ratio)

    # 2. MSI-based Mn indicators
    # Sentinel-2 band order: B1(443), B2(490), B3(560), B4(665), B5(705), B6(740), B7(783), B8(842), B8A(865), B9(945), B11(1610), B12(2190)

    # Mn-related plant stress indicators
    mn_ndvi = (msi_mean[7] - msi_mean[3]) / \
        (msi_mean[7] + msi_mean[3] + 1e-9)  # NIR-Red / NIR+Red
    mn_gndvi = (msi_mean[7] - msi_mean[2]) / (msi_mean[7] +
                                              msi_mean[2] + 1e-9)  # NIR-Green / NIR+Green

    # Effect of Mn on chlorophyll formation
    chlorophyll_mn = msi_mean[4] / (msi_mean[3] + 1e-9)  # Red-edge / Red
    chl_ratio = (msi_mean[4] - msi_mean[3]) / \
        (msi_mean[2] + 1e-9)  # (RE-R) / G

    features.extend([mn_ndvi, mn_gndvi, chlorophyll_mn, chl_ratio])

    # 3. Blue/Green entropy (Mn sensitivity indicators)
    def safe_entropy(val):
        val = np.abs(val) + 1e-9
        p = val / (val.sum() + 1e-9)
        return -(p * np.log(p + 1e-9)).sum()

    msi_blue_ent = safe_entropy(msi_mean[1:2])  # B2 entropy
    msi_green_ent = safe_entropy(msi_mean[2:3])  # B3 entropy
    msi_visible_ent = safe_entropy(msi_mean[1:4])  # B2-B3-B4 entropy

    features.extend([msi_blue_ent, msi_green_ent, msi_visible_ent])

    # 4. Mn-specific color indicators
    # Mn deficiency yellowing
    yellow_index = (msi_mean[2] + msi_mean[1]) / \
        (msi_mean[3] + 1e-9)  # (Green + Blue) / Red

    # Mn toxicity brown spots
    brown_index = (msi_mean[3] - msi_mean[2]) / \
        (msi_mean[1] + 1e-9)  # (Red - Green) / Blue

    # Purple-Red ratio (Mn oxide color)
    # Coastal / Red
    purple_red_ratio = msi_mean[0] / \
        (msi_mean[3] + 1e-9) if len(msi_mean) > 0 else 0

    features.extend([yellow_index, brown_index, purple_red_ratio])

    # 5. Red-Edge-related Mn indicators
    # Effect of Mn on photosynthesis
    re_position = 705 + (msi_mean[5] - msi_mean[3]) / \
        (msi_mean[5] - msi_mean[4] + 1e-9) * 35  # Red-edge position
    re_amplitude = msi_mean[4] - \
        (msi_mean[3] + msi_mean[5]) / 2  # Red-edge amplitude
    re_slope_mn = (msi_mean[5] - msi_mean[4]) / 35.0  # Red-edge slope

    features.extend([re_position, re_amplitude, re_slope_mn])

    # 6. SWIR-based Mn soil characteristics
    if len(msi_mean) >= 12:
        # Clay-Mn complex index
        clay_mn_index = msi_mean[10] / \
            (msi_mean[11] + 1e-9)  # B11/B12 (1610/2190)

        # Water-Mn interaction
        moisture_mn = (msi_mean[10] - msi_mean[11]) / \
            (msi_mean[10] + msi_mean[11] + 1e-9)  # NDWI-like

        features.extend([clay_mn_index, moisture_mn])
    else:
        features.extend([0, 0])

    return np.array(features, dtype=np.float32)


def fe_specific_features(hsi_mean, msi_mean):
    """Fe-specific features - use Fe oxide absorption bands"""
    features = []

    # Fe oxide characteristic bands (HSI)
    if len(hsi_mean) >= 198:
        # Fe oxide absorption in blue (400-500nm)
        blue_absorption = np.mean(hsi_mean[0:50])  # 400-500nm
        # Fe absorption in NIR (850-900nm)
        nir_absorption = np.mean(hsi_mean[90:100])  # 850-900nm
        # Fe absorption ratio
        fe_ratio = blue_absorption / (nir_absorption + 1e-9)
        features.append(fe_ratio)

    # Fe-related indicators in MSI
    # Red/Blue ratio (Fe oxide color)
    red_blue_ratio = msi_mean[3] / (msi_mean[1] + 1e-9)  # Red/Blue
    features.append(red_blue_ratio)

    return np.array(features, dtype=np.float32)


def zn_specific_features(hsi_mean, msi_mean):
    """Zn-specific features"""
    features = []

    # Zn deficiency yellowing (Green/Red increase)
    chlorosis_index = msi_mean[2] / (msi_mean[3] + 1e-9)  # Green/Red
    features.append(chlorosis_index)

    # NIR/Red ratio (related to plant health)
    ndvi_like = (msi_mean[7] - msi_mean[3]) / \
        (msi_mean[7] + msi_mean[3] + 1e-9)
    features.append(ndvi_like)

    return np.array(features, dtype=np.float32)


def spectral_indices_features(hsi_mean, msi_mean):
    """Add various spectral indices"""
    features = []

    # Soil-related indices
    # Soil Brightness Index
    sbi = np.sqrt((msi_mean[3]**2 + msi_mean[7]**2) / 2)  # Red, NIR
    features.append(sbi)

    # Soil Color Index
    sci = (msi_mean[3] - msi_mean[2]) / (msi_mean[3] + msi_mean[2] + 1e-9)
    features.append(sci)

    # Clay Minerals Ratio (SWIR bands)
    if len(msi_mean) >= 12:
        cmr = msi_mean[10] / (msi_mean[11] + 1e-9)  # SWIR1/SWIR2
        features.append(cmr)
    else:
        features.append(0)

    # Normalized Difference Vegetation Index
    ndvi = (msi_mean[7] - msi_mean[3]) / (msi_mean[7] + msi_mean[3] + 1e-9)
    features.append(ndvi)

    # Enhanced Vegetation Index
    evi = 2.5 * ((msi_mean[7] - msi_mean[3]) /
                 (msi_mean[7] + 6 * msi_mean[3] - 7.5 * msi_mean[1] + 1))
    features.append(evi)

    return np.array(features, dtype=np.float32)


# Sentinel-2 MSI band indices (total 12 bands: B10 omitted version)
B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 = range(12)


def s_specific_features_msi(msi_mean: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    msi_mean : np.ndarray shape (12,)
        Sentinel-2 12-band average reflectance (0-1 scale)

    Returns
    -------
    np.ndarray shape (8,)
        [SWIR_ratio, ND_s_Swir, PRI, RE_slope,
         sg1_swir, depth_2190, area_2190, NDMI, PSRI]
    ------------------------------------------------------------------
    * SWIR_ratio   : B12(2190)/B11(1610)  (clay·sulfate ↑ → ratio ↑)
    * ND_s_Swir    : (B11-B12)/(B11+B12)  (B11 absorption vs B12)
    * PRI          : (B2-B3)/(B2+B3)      (photosynthesis stress = S deficiency)
    * RE_slope     : (B6-B5)/35 nm        (red edge slope)
    * sg1_swir     : 1610-2190 nm 1st derivative RMS
    * depth_2190   : 2190 nm absorption depth (clay·alunite)
    * area_2190    : same absorption area
    * NDMI         : (B8-B11)/(B8+B11)    (water index, sulfate movement linked)
    * PSRI         : (B4-B3)/B7           (pigment-specific, S deficiency plants)
    """
    # basic indices -------------------------------------------------------
    # swir_ratio = msi_mean[B12] / (msi_mean[B11] + 1e-9)
    # nd_s_swir  = (msi_mean[B11] - msi_mean[B12]) / \
    #              (msi_mean[B11] + msi_mean[B12] + 1e-9)
    # pri        = (msi_mean[B2]  - msi_mean[B3])  / \
    #              (msi_mean[B2]  + msi_mean[B3]  + 1e-9)
    # re_slope   = (msi_mean[B6]  - msi_mean[B5])  / 35.0

    # 1610-2190 nm 1st derivative RMS ------------------------------------
    sg1_full = savgol_filter(msi_mean, 5, 2, deriv=1)   # win=5, poly=2
    sg1_swir = np.sqrt(((sg1_full[B11:B12+1])**2).mean())

    # 2190 nm absorption depth·area (continuum removal) ---------------------------
    # simple line criterion: line connecting B11(1610)-B12(2190)
    ref_line = np.interp([1610, 2190], [1610, 2190],
                         [msi_mean[B11], msi_mean[B12]])
    depth_2190 = 1.0 - msi_mean[B12] / (ref_line[1] + 1e-9)
    # area is only one pixel, so it's the same scalar as depth; if it were multi-band, it would be an integral
    area_2190 = depth_2190

    # NDMI (water) ---------------------------------------------
    ndmi = (msi_mean[B8] - msi_mean[B11]) / \
           (msi_mean[B8] + msi_mean[B11] + 1e-9)

    # PSRI (plant
    psri = (msi_mean[B4] - msi_mean[B3]) / (msi_mean[B7] + 1e-9)

    return np.array([
        sg1_swir, depth_2190, area_2190,
        ndmi, psri], dtype=np.float32)


EPS = 1e-9
WL_START = 400          # nm (band 0)
WL_STEP = 10           # nm (after bad-band removal)


def _w2i(wl: float) -> int:
    """wavelength (nm) → band index (0-based)"""
    return max(0, min(197, int(round((wl - WL_START) / WL_STEP))))


def _band_depth(vec: np.ndarray, ctr_wl: float,
                left_wl: float, right_wl: float) -> float:
    """
    absorption depth (continuum-removed depth) at center wavelength (ctr_wl).
    simply: difference between observed reflectance and linear continuum / continuum.
    """
    i_c, i_l, i_r = map(_w2i, (ctr_wl, left_wl, right_wl))
    R_c = vec[i_c]
    # value of continuum on the line connecting two endpoints
    R_l, R_r = vec[i_l], vec[i_r]
    R_cont = np.interp(ctr_wl, [left_wl, right_wl], [R_l, R_r])
    return 1.0 - R_c / (R_cont + EPS)


def _slope(vec: np.ndarray, wl1: float, wl2: float) -> float:
    i1, i2 = map(_w2i, (wl1, wl2))
    return (vec[i2] - vec[i1]) / (wl2 - wl1 + EPS)

# ──────────────────────────────────────────────────────────────
# ❷ Zn·Mn-specific advanced features
# ──────────────────────────────────────────────────────────────


def mn_advanced_features(hsi_mean: np.ndarray) -> np.ndarray:
    """
    Return shape (10,):
      [Mn_blue_slope, Mn_green_slope, Mn_620_depth, Mn_vis_ratio,
       Zn_blue_depth, Zn_550_depth, Zn_blue_green_ratio,
       Zn_vis_slope, Zn_nir_ratio, Zn_mn_cross]
    """
    feats = []
    # ── Mn-related ───────────────────────────────────────────────
    # ① 450–520 nm, ② 520–600 nm 1st derivative (Mn deficiency → two slopes ↓)
    feats.append(_slope(hsi_mean, 450, 500))   # Mn_blue_slope
    feats.append(_slope(hsi_mean, 520, 580))   # Mn_green_slope
    # ③ 620 nm absorption depth (Mn toxicity·oxide)
    feats.append(_band_depth(hsi_mean, 620, 580, 660))  # Mn_620_depth
    # ④ 550 / 660 nm ratio (green→brown reflectance change)
    feats.append(hsi_mean[_w2i(550)] /
                 (hsi_mean[_w2i(660)] + EPS))  # Mn_vis_ratio

    return np.array(feats, dtype=np.float32)


def zn_advanced_features(hsi_mean: np.ndarray) -> np.ndarray:
    """
    Return shape (10,):
      [Mn_blue_slope, Mn_green_slope, Mn_620_depth, Mn_vis_ratio,
       Zn_blue_depth, Zn_550_depth, Zn_blue_green_ratio,
       Zn_vis_slope, Zn_nir_ratio, Zn_mn_cross]
    """
    feats = []
    # ── Zn-related ───────────────────────────────────────────────
    # (Zn deficiency = chlorophyll↓, blue·green reflectance↑, Red-edge position variation, etc.)
    feats.append(_band_depth(hsi_mean, 440, 400, 480))  # Zn_blue_depth
    feats.append(_band_depth(hsi_mean, 550, 520, 580))  # Zn_550_depth
    # Zn_blue_green_ratio
    feats.append(hsi_mean[_w2i(520)] / (hsi_mean[_w2i(560)] + EPS))
    feats.append(_slope(hsi_mean, 680, 740))            # Zn_vis_slope
    feats.append(hsi_mean[_w2i(800)] /
                 (hsi_mean[_w2i(660)] + EPS))  # Zn_nir_ratio

    return np.array(feats, dtype=np.float32)


# ------------------------------------------------------------
# Parameters & helpers
# ------------------------------------------------------------
EPS = 1e-9
WL_START = 400     # nm  (index 0)
WL_STEP = 10      # nm  (400→410→…)
MAX_IDX = 197     # last index (400+10*197 = 2370 nm)


def wl2idx(wl: float) -> int:
    """wavelength (nm) → 0-based band index (clamped)"""
    return int(np.clip(round((wl - WL_START) / WL_STEP), 0, MAX_IDX))


def slope(vec, wl1, wl2):
    i1, i2 = wl2idx(wl1), wl2idx(wl2)
    return (vec[i2] - vec[i1]) / (wl2 - wl1 + EPS)


def cont_depth(vec, ctr, left, right):
    """continuum-removed absorption depth (1 − R_ctr / R_continuum)"""
    ic, il, ir = wl2idx(ctr), wl2idx(left), wl2idx(right)
    Rl, Rr = vec[il], vec[ir]
    R_cont = np.interp(ctr, [left, right], [Rl, Rr])
    return 1.0 - vec[ic] / (R_cont + EPS)


def band_ratio(vec, wl_num, wl_den):
    return vec[wl2idx(wl_num)] / (vec[wl2idx(wl_den)] + EPS)


def zn_hsi_features(hsi_vec: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    hsi_vec : (198,)  after bad-band removal, spectrum (400-2370 nm)

    Returns
    -------
    np.ndarray (12,)  – Zn sensitive index vector
    Feature order:
      [CRI550, BlueDepth, GreenDepth, BlueGreenSlope,
       RE_slope, RE_position, Datt705, DerivMax, DerivMean,
       BlueRatio, GreenRatio, NIR_RedRatio]
    """
    feats = []
    # ── 1. carotenoid / chlorophyll indicators ───────────────────────────
    # CRI550 (1/R705 − 1/R665) ∝ carotenoid
    cri550 = (1/(hsi_vec[wl2idx(705)] + EPS) -
              1/(hsi_vec[wl2idx(665)] + EPS))
    feats.append(cri550)

    # ── 2. Continuum-removed depth (Blue 440, Green 550) ───────
    feats.append(cont_depth(hsi_vec, 440, 400, 480))   # BlueDepth
    feats.append(cont_depth(hsi_vec, 550, 520, 580))   # GreenDepth

    # ── 3. Blue-Green slope & ratio ───────────────────────────
    feats.append(slope(hsi_vec, 430, 560))             # BlueGreenSlope

    # ── 4. Red-edge characteristics ─────────────────────────────────────
    re_slope = slope(hsi_vec, 705, 740)                # RE_slope
    feats.append(re_slope)

    # linear 3-point Red-edge position(λ_RE) [Guyot & Baret]
    re_pos = 705 + (hsi_vec[wl2idx(740)] - hsi_vec[wl2idx(665)]) / \
        (hsi_vec[wl2idx(740)] - hsi_vec[wl2idx(705)] + EPS) * 35
    feats.append(re_pos)

    # Datt705 ratio (783 nm / 705 nm)
    feats.append(band_ratio(hsi_vec, 783, 705))        # Datt705

    # ── 5. 1st derivative statistics (680-750 nm) ────────────────────
    idx_680, idx_750 = wl2idx(680), wl2idx(750)
    deriv = np.diff(hsi_vec[idx_680:idx_750+1]) / WL_STEP
    feats.append(np.max(deriv))                        # DerivMax
    feats.append(np.mean(deriv))                       # DerivMean

    # ── 6. simple band ratio (Blue/Red, Green/Red, NIR/Red) ───────
    feats.append(band_ratio(hsi_vec, 450, 660))        # BlueRatio
    feats.append(band_ratio(hsi_vec, 550, 660))        # GreenRatio
    feats.append(band_ratio(hsi_vec, 800, 660))        # NIR_RedRatio

    return np.array(feats, dtype=np.float32)


def zn_hsi_features_sg(hsi_vec):
    # ① 5 bands (SG window=5) 2nd Poly smoothing
    sm = savgol_filter(hsi_vec, 5, 2)
    # ② 1st derivative
    deriv = savgol_filter(hsi_vec, 5, 2, deriv=1)     # dR/dλ
    # ③ Zn sensitive parameters
    #    – max dR/dλ  (680–750 nm) : red-edge slope
    idx1, idx2 = wl2idx(680), wl2idx(750)
    dmax = deriv[idx1:idx2+1].max()
    dmean = deriv[idx1:idx2+1].mean()

    #    – derivative slope ratio (Blue vs Green)
    ds_blue = slope(sm, 430, 480)      # dR/dλ 430-480
    ds_green = slope(sm, 520, 580)
    slope_ratio = ds_blue / (ds_green + EPS)

    #    – CRI550 (smoothed value)
    cri550 = (1/(sm[wl2idx(705)]+EPS) - 1/(sm[wl2idx(665)]+EPS))

    return np.array([dmax, dmean, slope_ratio, cri550], dtype=np.float32)


# selected MSI bands → texture calculation (e.g., Red, NIR)
TEXTURE_BANDS = [B3, B7]                     # Green(560 nm), NIR(783 nm)
GLCM_DISTANCES = (1, 2)                      # 1·2 pixel distance
GLCM_ANGLES = (0, np.pi/4, np.pi/2, 3*np.pi/4)


def spatial_texture_features(msi_patch: np.ndarray,
                             mask_patch: np.ndarray) -> np.ndarray:
    """
    msi_patch : (12, H, W) Sentinel-2 patch (resized)
    mask_patch: (1, H, W)  valid pixel mask (True=valid)
    return     : (n_features,)
    """
    feats = []

    # ── 1. GLCM texture ────────────────────────────────────────
    # rescale to 8-bit & apply mask
    def _norm8(a): return np.clip((a - a.min()) /
                                  (a.ptp() + 1e-9) * 255, 0, 255).astype(np.uint8)

    for b in TEXTURE_BANDS:
        img = np.where(mask_patch[0], msi_patch[b], np.nan)
        if np.isnan(img).all():
            # contrast, homogeneity, entropy, dissimilarity
            feats.extend([0]*4)
            continue

        img8 = _norm8(np.nan_to_num(img, nan=np.nanmean(img)))
        glcm = graycomatrix(img8,
                            distances=GLCM_DISTANCES,
                            angles=GLCM_ANGLES,
                            levels=256, symmetric=True, normed=True)
        feats.append(graycoprops(glcm, 'contrast').mean())
        feats.append(graycoprops(glcm, 'homogeneity').mean())
        feats.append(graycoprops(glcm, 'dissimilarity').mean())
        # simple entropy
        p = glcm / (glcm.sum() + 1e-9)
        feats.append(-(p * np.log(p + 1e-9)).sum())

    # ── 2. Edge/Gradient statistics ────────────────────────────────
    # NIR band-based
    nir = np.where(mask_patch[0], msi_patch[B8], 0.0)
    grad_mag = np.hypot(sobel(nir, axis=0), sobel(nir, axis=1))
    feats.extend([
        grad_mag.mean(),
        grad_mag.std(),
        np.percentile(grad_mag, 90)
    ])

    return np.array(feats, dtype=np.float32)


def build_feature_matrix_and_labels(indices: np.ndarray,
                                    gt_df,
                                    dataset_dir: Path,
                                    label_mean: np.ndarray,
                                    label_std: np.ndarray,
                                    is_train: bool = True,
                                    rotate_aug: bool = False,
                                    hsi_size: tuple = (4, 4),
                                    msi_size: tuple = (11, 11),
                                    hsi_fft_ratio: float = 0.2,
                                    msi_fft_ratio: float = 0.2,
                                    msi_mean_fft_ratio: float = 0.2,
                                    msi_max_fft_ratio: float = 0.2,
                                    hsi_mean_fft_ratio: float = 0.,
                                    hsi_max_fft_ratio: float = 0.,
                                    add_extra_features: bool = True,
                                    use_fe_features: bool = True,
                                    use_zn_features: bool = False,
                                    use_s_features: bool = False,
                                    use_mn_advanced_features: bool = False,
                                    use_zn_advanced_features: bool = False,
                                    use_spatial_features: bool = False,
                                    full_model_type: str = 'large',
                                    target_parameters: list = None,
                                    **kwargs
                                    ):
    """
    Satellite HSI·MSI data to create feature matrix (X) and label (Y).

    Parameters
    ----------
    rotate_aug : bool, optional (default=False)
        • True  → 90·180·270° rotation augmentation (total 4x)  
        • False → same as original
    hsi_fft_ratio : float, optional (default=0.2)
        HSI-related features use fft_low_k k ratio (0 → exclude FFT features)
    msi_fft_ratio : float, optional (default=0.2)
        MSI-related features use fft_low_k k ratio (0 → exclude FFT features)
    msi_mean_fft_ratio : float, optional (default=0.2)
        MSI mean FFT use k ratio (0 → exclude FFT features)
    msi_max_fft_ratio : float, optional (default=0.2)
        MSI max FFT use k ratio (0 → exclude FFT features)
    hsi_mean_fft_ratio : float, optional (default=0.2)
        HSI mean FFT use k ratio (0 → exclude FFT features)
    hsi_max_fft_ratio : float, optional (default=0.2)
        HSI max FFT use k ratio (0 → exclude FFT features)
    add_extra_features : bool, optional (default=True)
        use specialized features (Mn-specific features are always included)
            use_fe_features : bool, optional (default=True)
        Fe-specific features use 
    use_zn_features : bool, optional (default=False)
        Zn-specific features use 
    use_s_features : bool, optional (default=False)
        spectral index features use 
    """
    # ───── path setup and initialization ───────────────────────────────────
    base_dir = dataset_dir / ('train' if is_train else 'test')
    hsi_satellite_dir = base_dir / 'hsi_satellite'
    msi_satellite_dir = base_dir / 'msi_satellite'

    if not is_train:
        indices = np.arange(len(glob(f"{msi_satellite_dir}/*.npz")))

    if is_train:
        # If target_parameters is not provided, use default values
        if target_parameters is None:
            target_parameters = ['Fe', 'Zn', 'B', 'Cu', 'S', 'Mn']
        gt_df_labels = gt_df[target_parameters]

    features_list, labels_list = [], []

    def _extract_hsi_features(hsi_res, hsi_mask_res):
        """extract HSI-related features"""
        masked_hsi = np.ma.MaskedArray(hsi_res, mask=hsi_mask_res)
        orig_hsi_mean = masked_hsi.mean(axis=(1, 2)).filled(0)
        hsi_mean = remove_bad_bands_hsi(orig_hsi_mean)
        hsi_d1 = compute_derivatives(hsi_mean, 1)

        hsi_max = masked_hsi.max(axis=(1, 2)).filled(0)
        hsi_max = remove_bad_bands_hsi(hsi_max)

        # HSI SVD processing
        hsi_s0 = []
        for c in range(hsi_res.shape[0]):
            try:
                _, s, _ = svd(hsi_res[c], full_matrices=False)
            except LinAlgError:
                s = np.zeros(1)
            hsi_s0.append(s[0] if len(s) > 0 else 0)

        return orig_hsi_mean, hsi_mean, hsi_max, hsi_d1, np.array(hsi_s0)

    def _extract_msi_features(msi_res, msi_mask_res):
        """extract MSI-related features"""
        masked_msi = np.ma.MaskedArray(msi_res, mask=msi_mask_res)
        msi_mean = masked_msi.mean(axis=(1, 2)).filled(0)
        msi_max = masked_msi.max(axis=(1, 2)).filled(0)
        msi_d1 = compute_derivatives(msi_mean, 1)

        # MSI SVD processing
        msi_s0 = []
        msvd_ratio = []
        for c in range(msi_res.shape[0]):
            try:
                _, s, _ = svd(msi_res[c], full_matrices=False)
            except LinAlgError:
                s = np.zeros(2)

            msi_s0.append(s[0] if len(s) > 0 else 0)
            msvd_ratio.append(s[0]/s[1] if len(s) >= 2 and s[1] > 1e-9 else 0)

        return msi_mean, msi_max, msi_d1, np.array(msi_s0), np.array(msvd_ratio)

    def _apply_fft_if_enabled(data_array, ratio, label=""):
        """apply FFT features only if FFT ratio is not 0"""
        if ratio > 0 and len(data_array) > 0:
            k = max(1, int(len(data_array) * ratio))
            real_part = np.real(fft_low_k(data_array, k=k))
            imag_part = np.imag(fft_low_k(data_array, k=k))
            return real_part, imag_part
        else:
            return np.array([]), np.array([])

    rng = np.random.default_rng()

    # ─────────────────────────────────────────────────────────────
    # helper : add Gaussian noise
    # ─────────────────────────────────────────────────────────────
    def add_noise(arr, sigma_range=(0.005, 0.02)):
        sigma = rng.uniform(*sigma_range)
        return np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 1).astype(arr.dtype)

    # ─────────────────────────────────────────────────────────────
    # main : rotation·flip·noise augmentation
    # ─────────────────────────────────────────────────────────────
    def gen_augmented_variants(hsi, hsi_m, msi, msi_m, with_noise=True):
        """
        Parameters
        ----------
        hsi, msi : np.ndarray  (C,H,W)
        hsi_m, msi_m : bool mask (C,H,W)  (True=invalid)
        with_noise : bool  (True → return both noisy and noiseless)

        Yields
        ------
        tuple(hsi_aug, hsi_mask_aug, msi_aug, msi_mask_aug)
        """
        # 8-way spatial transformation (D4)
        for k in (0, 1, 2, 3):                       # 0°,90°,180°,270°
            h_r = np.rot90(hsi,  k, axes=(1, 2))
            hm_r = np.rot90(hsi_m, k, axes=(1, 2))
            m_r = np.rot90(msi,  k, axes=(1, 2))
            mm_r = np.rot90(msi_m, k, axes=(1, 2))

            # H flip off/on (two)
            for flip in (False, True):
                if flip:
                    h_f = np.flip(h_r,  axis=2)     # left↔right
                    hm_f = np.flip(hm_r, axis=2)
                    m_f = np.flip(m_r,  axis=2)
                    mm_f = np.flip(mm_r, axis=2)
                else:
                    h_f, hm_f, m_f, mm_f = h_r, hm_r, m_r, mm_r

                # noise off/on (two)
                if with_noise:
                    yield h_f, hm_f, m_f, mm_f                     # original
                    yield add_noise(h_f), hm_f, add_noise(m_f), mm_f  # noise
                else:
                    yield h_f, hm_f, m_f, mm_f

    # ─────────────────────────────────────────────────────────────
    #  new downsampling augmentation (before resize_patch call)
    # ─────────────────────────────────────────────────────────────
    def gen_downsample_variants(hsi_d, hsi_m, msi_d, msi_m, msi_target, full_model_type):
        """create downsampled variants before resize_patch call.

        HSI  : 1×1 to (target−1)×(target−1)
        MSI  : 2×2 to (target−1)×(target−1)  (example requirement)

        Yields
        ------
        tuple(hsi_var, hsi_mask_var, msi_var, msi_mask_var)
        """
        target_size = msi_target[0]  # assume square

        if full_model_type == 'large' or full_model_type == 'base':
            return  # no downsampling

        # medium → 3, small → 2
        # min_size = 3 if full_model_type == 'medium' else 2
        min_size = 3  # medium, small downsample to 3

        # MSI downsampling variants-------------------------------------------------
        for s in range(target_size - 1, min_size - 1, -1):
            if s < min_size:
                break
            try:
                m_ds, m_m_ds = resize_patch(msi_d, msi_m, s, s)
                yield hsi_d, hsi_m, m_ds, m_m_ds
            except Exception:
                continue

    def _extract_features(hsi_res, hsi_mask_res, msi_res, msi_mask_res):
        """extract 1-D feature vector from one patch"""
        # extract HSI features
        orig_hsi_mean, hsi_mean, hsi_max, hsi_d1, hsi_s0 = _extract_hsi_features(
            hsi_res, hsi_mask_res)

        # extract MSI features
        msi_mean, msi_max, msi_d1, msi_s0, msvd_ratio = _extract_msi_features(
            msi_res, msi_mask_res)

        # start with basic features
        feat_parts = [msi_d1, msi_max]

        # MSI FFT features (only if ratio > 0)
        if msi_fft_ratio > 0:
            # msvd_ratio FFT
            mfft_real, mfft_imag = _apply_fft_if_enabled(
                msvd_ratio, msi_fft_ratio)
            if len(mfft_real) > 0:
                feat_parts.extend([mfft_real, mfft_imag])

            # msi_s0 FFT
            msi_s0_real, msi_s0_imag = _apply_fft_if_enabled(
                msi_s0, msi_fft_ratio)
            if len(msi_s0_real) > 0:
                feat_parts.extend([msi_s0_real, msi_s0_imag])

            # msi_d1 FFT
            msi_d1_real, msi_d1_imag = _apply_fft_if_enabled(
                msi_d1, msi_fft_ratio)
            if len(msi_d1_real) > 0:
                feat_parts.extend([msi_d1_real, msi_d1_imag])

        if msi_mean_fft_ratio > 0:
            # msi_mean FFT
            msi_mean_real, msi_mean_imag = _apply_fft_if_enabled(
                msi_mean, msi_mean_fft_ratio)
            if len(msi_mean_real) > 0:
                feat_parts.extend([msi_mean_real, msi_mean_imag])

        if msi_max_fft_ratio > 0:
            # msi_max FFT
            msi_max_real, msi_max_imag = _apply_fft_if_enabled(
                msi_max, msi_max_fft_ratio)
            if len(msi_max_real) > 0:
                feat_parts.extend([msi_max_real, msi_max_imag])

        # HSI FFT features (only if ratio > 0)
        if hsi_fft_ratio > 0:
            hsi_s0_real, hsi_s0_imag = _apply_fft_if_enabled(
                hsi_s0, hsi_fft_ratio)
            if len(hsi_s0_real) > 0:
                feat_parts.extend([hsi_s0_real, hsi_s0_imag])

        # HSI mean FFT features (only if ratio > 0)
        if hsi_mean_fft_ratio > 0:
            hsi_mean_real, hsi_mean_imag = _apply_fft_if_enabled(
                hsi_mean, hsi_mean_fft_ratio)
            # hsi_mean_real, hsi_mean_imag = _apply_fft_if_enabled_hsi(
            #     hsi_mean, hsi_mean_fft_ratio)
            if len(hsi_mean_real) > 0:
                feat_parts.extend([hsi_mean_real, hsi_mean_imag])

        if hsi_max_fft_ratio > 0:
            hsi_max_real, hsi_max_imag = _apply_fft_if_enabled(
                hsi_max, hsi_max_fft_ratio)
            # hsi_max_real, hsi_max_imag = _apply_fft_if_enabled_hsi(
            #     hsi_max, hsi_max_fft_ratio)
            if len(hsi_max_real) > 0:
                feat_parts.extend([hsi_max_real, hsi_max_imag])

        if add_extra_features:
            # Mn-specific features
            mn_spec1 = mn_specific_features(msi_mean)
            # mn_spec2 = mn_specific_features(hsi_mean, msi_max)
            feat_parts.extend([mn_spec1])

        # Fe-specific features
        if use_fe_features:
            fe_spec1 = fe_specific_features(orig_hsi_mean, msi_mean)
            feat_parts.extend([fe_spec1])

        # Zn-specific features
        if use_zn_features:
            zn_spec1 = zn_specific_features(orig_hsi_mean, msi_mean)
            feat_parts.extend([zn_spec1])

        # spectral index features
        if use_s_features:
            s_spec = s_specific_features_msi(msi_mean)
            feat_parts.append(s_spec)

        # Mn-specific advanced features
        if use_mn_advanced_features:
            mn_adv = mn_advanced_features(hsi_mean)
            feat_parts.append(mn_adv)

        # Zn-specific advanced features
        if use_zn_advanced_features:
            zn_adv = zn_advanced_features(orig_hsi_mean)
            feat_parts.append(zn_adv)

        # spatial texture features
        if use_spatial_features:
            spatial_feat = spatial_texture_features(msi_res, msi_mask_res)
            feat_parts.append(spatial_feat)

        feat = np.concatenate(feat_parts, dtype=np.float32)
        return np.nan_to_num(feat)

    # ───── main loop ──────────────────────────────────────────
    for idx in tqdm(indices, desc="Building Satellite (HSI+MSI) features"):
        try:
            # load and resize data
            hsi_data, hsi_mask = _load_data(
                hsi_satellite_dir / f"{idx:04}.npz")
            msi_data, msi_mask = _load_data(
                msi_satellite_dir / f"{idx:04}.npz")

            # ── downsampling augmentation ───────────────────────────
            variant_pairs = [(hsi_data, hsi_mask, msi_data, msi_mask)]

            # if is_train:  # augmentation only for training
            #     variant_pairs.extend(gen_downsample_variants(
            #         hsi_data, hsi_mask, msi_data, msi_mask, msi_size, full_model_type))

            if is_train:
                log_label = gt_df_labels.iloc[idx].values.astype(np.float32)
                norm_lbl = (log_label - label_mean) / label_std

            for h_d, h_m_d, m_d, m_m_d in variant_pairs:
                # restore variant to target size
                hsi_res, hsi_m_res = resize_patch(h_d, h_m_d, *hsi_size)
                msi_res, msi_m_res = resize_patch(m_d, m_m_d, *msi_size)

                features_list.append(_extract_features(
                    hsi_res, hsi_m_res, msi_res, msi_m_res))

                if is_train:
                    labels_list.append(norm_lbl)

                    # rotation·flip·noise augmentation (original)
                    if rotate_aug:
                        for hsi_a, hsi_ma, msi_a, msi_ma in gen_augmented_variants(
                                hsi_res, hsi_m_res, msi_res, msi_m_res, with_noise=False):
                            features_list.append(_extract_features(
                                hsi_a, hsi_ma, msi_a, msi_ma))
                            labels_list.append(norm_lbl)

        except Exception as e:
            print(f"[ERROR] idx {idx}: {e} — skipping.")
            continue

    X = np.stack(features_list, axis=0)
    Y = np.stack(labels_list, axis=0) if is_train else None
    return X, Y


def _load_data(file_path):
    """data load helper function"""
    with np.load(file_path) as npz:
        data = npz['data']
        mask = npz['mask'].astype(bool)
    return data, mask
