MNSPEC_NAMES = [
    # mn_specific_features(msi_mean)
    "Ratio between avg. reflectance at 490 and 560 nm",          # 0
    "Normalized Difference Vegetation Index",              # 1
    "Green Normalized Difference Vegetation Index",             # 2
    "Chlorophyll",       # 3
    "Chlorophyll Ratio",            # 4
    "Entropy at 490nm ",         # 5
    "Entropy at 560nm",        # 6
    "Entropy of 490, 560, and 665 nm",      # 7
    "yellow_index",         # 8
    "brown_index",          # 9
    "Ratio between avg.reflectance at 443nm and 665 nm",     # 10
    "Position of red edge",          # 11
    "Amplitude of red edge",         # 12
    "Slope of red edge",          # 13
    "Ratio between avg. reflectance at 1610 and 2190 nm",        # 14
    "Normalized Difference Water Index",          # 15  ← your example
]
FESPEC_NAMES = [
    # fe_specific_features(hsi_mean, msi_mean)
    # (Assumes hsi_mean has length >= 198, which matches your spec of 2 features.)
    "Fe ratio",             # 0
    "Ratio between avg. reflectance at 665 and 490 nm",       # 1
]
ZNSPEC_NAMES = [
    # zn_specific_features(hsi_mean, msi_mean)
    "Ratio between avg. reflectance at 665 and 490 nm",      # 0
    "Normalized Difference Vegetation Index",            # 1
]
SSPEC_NAMES = [
    # s_specific_features_msi(msi_mean)
    # (Function currently returns 5 values; docstring lists more, but only these are computed.)
    "Root mean squared of 1st derivative of avg. reflectance at 1610 and 2190 nm",             # 0
    "Continuum removal at 2190nm",           # 1
    "Continuum removal at 2190nm",            # 2
    "Normalized Difference Moisture Index",                 # 3
    "Plant Senescence Reflectance Index",                 # 4
]
MNADV_NAMES = [
    # mn_advanced_features(hsi_mean)
    # (Function returns 4 values in your snippet.)
    "Slope between 450 and 500nm (hyperspectral)",        # 0
    "Slope between 520 and 580nm (hyperspectral)",       # 1
    "Continuum removal at 620nm (hyperspectral)",         # 2
    "Ratio between 550 and 600nm (hyperspectral)",         # 3
]
large_spec = [
    ("1st derivative of MSI avg. reflectance", 12),
    ("MSI max. reflectance", 12),
    ("real part of FFT of ratio of 1st/2nd diag of MSI SVD", 2),
    ("imag part of FFT of ratio of 1st/2nd diag of MSI SVD", 2),
    ("real part of FFT of 1st diag of MSI SVD", 2),
    ("imag part of FFT of 1st diag of MSI SVD", 2),
    ("real part of FFT of 1st derivative of MSI avg. reflectance", 2),
    ("imag part of FFT of 1st derivative of MSI avg. reflectance", 2),
    ("real part of FFT of MSI avg. reflectance", 2),
    ("imag part of FFT of MSI avg. reflectance", 2),
    ("real part of FFT of MSI max. reflectance", 2),
    ("imag part of FFT of MSI max. reflectance", 2),
    ("real part of FFT of 1st diag of HSI SVD", 23),
    ("imag part of FFT of 1st diag of HSI SVD", 23),
    ("mnspec", 16),  # 106
    ("fespec", 2),  # 108
    ("sspec", 5),  # 113
]
medium_spec = [
    ("1st derivative of MSI avg. reflectance", 12),
    ("MSI max. reflectance", 12),
    ("real part of FFT of ratio of 1st/2nd diag of MSI SVD", 2),
    ("imag part of FFT of ratio of 1st/2nd diag of MSI SVD", 2),
    ("real part of FFT of MSI 1st diag of SVD", 2),
    ("imag part of FFT of MSI 1st diag of SVD", 2),
    ("real part of FFT of 1st derivative of MSI avg. reflectance", 2),
    ("imag part of FFT of 1st derivative of MSI avg. reflectance", 2),
    ("real part of FFT of MSI avg. reflectance", 2),
    ("imag part of FFT of MSI avg. reflectance", 2),
    ("real part of FFT of MSI max. reflectance", 2),
    ("imag part of FFT of MSI max. reflectance", 2),
    ("real part of FFT of 1st diag of HSI SVD", 46),
    ("imag part of FFT of 1st diag of HSI SVD", 46),
    ("mnspec", 16),
    ("fespec", 2),
    ("sspec", 5),
    ("mnadv", 4),
]
# total dims (for sanity): 163
small_spec = [
    ("1st derivative of MSI avg. reflectance", 12),
    ("MSI max. reflectance", 12),
    ("real part of FFT of ratio of 1st/2nd diag of MSI SVD", 2),
    ("imag part of FFT of ratio of 1st/2nd diag of MSI SVD", 2),
    ("real part of FFT of 1st diag of MSI SVD", 2),
    ("imag part of FFT of 1st diag of MSI SVD", 2),
    ("real part of FFT of 1st derivative of MSI avg. reflectance", 2),
    ("imag part of FFT of 1st derivative of MSI avg. reflectance", 2),
    ("real part of FFT of MSI avg. reflectance", 1),
    ("imag part of FFT of MSI avg. reflectance", 1),
    ("real part of FFT of MSI max. reflectance", 1),
    ("imag part of FFT of MSI max. reflectance", 1),
    ("real part of FFT of 1st diag of HSI SVD", 69),
    ("imag part of FFT of 1st diag of HSI SVD", 69),
    ("mnspec", 16),
    ("fespec", 2),
    ("znspec", 2),
    ("sspec", 5),
]
# --------------------------------------------------------------------
wavelengths = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190]


def build_index_map(spec):
    """Return two dicts:
         idx2feat[i]  ->  'feature_name[k]'
         feat2idx[(feature_name, k)] -> i
       where k is the *in-feature* index (zero-based)."""
    idx2feat, feat2idx = {}, {}
    idx = 0
    for name, length in spec:
        if name != '1st derivative of MSI avg. reflectance' and name != 'MSI max. reflectance':
            if name == 'mnspec':
                for k in range(length):
                    idx2feat[idx] = MNSPEC_NAMES[k]
                    idx += 1
            elif name == 'fespec':
                for k in range(length):
                    idx2feat[idx] = FESPEC_NAMES[k]
                    idx += 1
            elif name == 'znspec':
                for k in range(length):
                    idx2feat[idx] = ZNSPEC_NAMES[k]
                    idx += 1
            elif name == 'sspec':
                for k in range(length):
                    idx2feat[idx] = SSPEC_NAMES[k]
                    idx += 1
            elif name == 'mnadv':
                for k in range(length):
                    idx2feat[idx] = MNADV_NAMES[k]
                    idx += 1
            else:
                for k in range(length):
                    idx2feat[idx] = f"Component no. {k+1} of {name}"
                    feat2idx[(name, k)] = idx
                    idx += 1
        else:
            for k in range(length):
                idx2feat[idx] = f"{name}|{wavelengths[k]}nm"
                feat2idx[(name, k)] = idx
                idx += 1
    return idx2feat, feat2idx


idx2feat_L, feat2idx_L = build_index_map(large_spec)
idx2feat_M, feat2idx_M = build_index_map(medium_spec)
idx2feat_S, feat2idx_S = build_index_map(small_spec)

idx2feat_dict = {}
idx2feat_dict['large'] = idx2feat_L
idx2feat_dict['medium'] = idx2feat_M
idx2feat_dict['small'] = idx2feat_S
