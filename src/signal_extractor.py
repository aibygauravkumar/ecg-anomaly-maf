"""
src/signal_extractor.py — Extract ECG signal from real medical ECG reports

Two scenarios handled automatically:
  1. Full 12-lead ECG report (like the image you uploaded)
     → Detects patient header, finds rhythm strip at bottom, ignores everything else

  2. Single lead strip (clean screenshot or cropped image)
     → Processes the whole image directly

Pipeline:
  Step 1: Detect report type (full report vs single lead)
  Step 2: If full report → crop to rhythm strip only
  Step 3: Remove ECG grid background (red/pink squares)
  Step 4: Extract signal trace (darkest line)
  Step 5: Detect R-peaks and segment one clean beat
  Step 6: Resample to 187 points
"""

import numpy as np
import neurokit2 as nk
from PIL import Image, ImageDraw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Detect whether this is a full report or a single lead
# ─────────────────────────────────────────────────────────────────────────────

def detect_report_type(image: Image.Image) -> str:
    """
    Returns 'full_report' or 'single_lead'.

    A full ECG report has:
    - Large image (width > 800px typically)
    - Landscape orientation (wider than tall)
    - Lots of red/pink grid pixels (ECG paper)
    - Multiple rows of waveforms
    """
    w, h     = image.size
    aspect   = w / h

    img_rgb  = np.array(image.convert("RGB"), dtype=np.float32)
    R, G, B  = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    red_pct  = ((R > 140) & (R - G > 25) & (R - B > 25)).mean()

    # Full report: landscape, large, lots of grid
    if aspect > 1.2 and w > 600 and red_pct > 0.10:
        return "full_report"
    return "single_lead"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — For full reports: locate patient header and rhythm strip
# ─────────────────────────────────────────────────────────────────────────────

def locate_ecg_regions(image: Image.Image) -> dict:
    """
    Scan the image row by row to find:
      header_end   — where the patient info section ends
      ecg_start    — where the ECG grid begins
      ecg_end      — where the ECG grid ends (before footer)
      rhythm_start — where the bottom rhythm strip begins

    Method: count red/pink pixels per row.
    Header rows have very few (it's just white with black text).
    ECG rows have many (covered with grid).
    Footer rows have very few again.
    The rhythm strip is the bottom ~22% of the ECG area.
    """
    img_rgb      = np.array(image.convert("RGB"), dtype=np.float32)
    height, width = img_rgb.shape[:2]

    R, G, B      = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    is_grid      = (R > 140) & (R - G > 25) & (R - B > 25)

    # Fraction of red pixels per row
    row_red_density = is_grid.mean(axis=1)   # shape (height,)

    # Rows where the ECG grid is present
    GRID_THRESHOLD = 0.04
    grid_rows = np.where(row_red_density > GRID_THRESHOLD)[0]

    if len(grid_rows) < 20:
        # Cannot detect grid — treat as single lead
        return {
            "ecg_start":    0,
            "ecg_end":      height,
            "rhythm_start": int(height * 0.75),
            "rhythm_end":   height,
            "header_end":   0,
        }

    ecg_start  = int(grid_rows[0])
    ecg_end    = int(grid_rows[-1])
    ecg_height = ecg_end - ecg_start

    # Standard 12-lead layout:
    # 3 rows of 4 leads → each takes ~25% of ECG height
    # 1 rhythm strip    → bottom ~22% of ECG height
    # We find the rhythm strip by looking for a horizontal gap
    # (thin black separator line) in the bottom half

    # Look for the separator line: a row with almost no grid and no black trace
    # i.e., a nearly white row inside the ECG area
    gray         = np.array(image.convert("L"), dtype=np.float32)
    row_darkness = gray[ecg_start:ecg_end].mean(axis=1)   # bright rows = white/separator
    row_redness  = row_red_density[ecg_start:ecg_end]

    # Search in bottom half of ECG area for the separator
    search_start = int(ecg_height * 0.60)
    search_end   = int(ecg_height * 0.85)

    best_separator = ecg_start + int(ecg_height * 0.78)   # default

    # A separator row is: bright (>200 mean) and NOT red (grid absent)
    for i in range(search_start, search_end):
        if row_darkness[i] > 210 and row_redness[i] < 0.02:
            best_separator = ecg_start + i
            break

    return {
        "ecg_start":    ecg_start,
        "ecg_end":      ecg_end,
        "rhythm_start": best_separator,
        "rhythm_end":   ecg_end,
        "header_end":   ecg_start,
    }


def crop_rhythm_strip(image: Image.Image, regions: dict) -> Image.Image:
    """
    Crop to just the rhythm strip region.
    Adds a small vertical padding so we do not clip the waveform edges.
    """
    w      = image.width
    pad    = 10
    top    = max(0,            regions["rhythm_start"] - pad)
    bottom = min(image.height, regions["rhythm_end"]   + pad)
    return image.crop((0, top, w, bottom))


def annotate_regions(image: Image.Image, regions: dict) -> Image.Image:
    """
    Draw colored boxes on the image showing detected regions.
    Used in the app to show the user what was found.
    """
    annotated = image.convert("RGB").copy()
    draw      = ImageDraw.Draw(annotated, "RGBA")

    w = image.width

    # Header region (red overlay)
    draw.rectangle(
        [0, 0, w, regions["header_end"]],
        fill=(255, 0, 0, 60), outline=(255, 0, 0, 200)
    )

    # ECG leads grid region (yellow overlay)
    draw.rectangle(
        [0, regions["ecg_start"], w, regions["rhythm_start"]],
        fill=(255, 200, 0, 40), outline=(200, 150, 0, 180)
    )

    # Rhythm strip (green overlay) — this is what we use
    draw.rectangle(
        [0, regions["rhythm_start"], w, regions["rhythm_end"]],
        fill=(0, 200, 0, 60), outline=(0, 180, 0, 220)
    )

    return annotated


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Remove the ECG grid background
# ─────────────────────────────────────────────────────────────────────────────

def remove_ecg_grid(image: Image.Image) -> Image.Image:
    """
    Remove red/pink ECG paper grid by replacing red pixels with white.
    Keeps only the dark signal trace.
    """
    img_rgb = np.array(image.convert("RGB"), dtype=np.float32)
    R, G, B = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    is_red  = (R > 140) & (R - G > 25) & (R - B > 25)
    brightness = (R + G + B) / 3
    is_light_gray = (brightness > 200) & (np.abs(R-G) < 15) & (np.abs(G-B) < 15)

    cleaned = img_rgb.copy()
    cleaned[is_red | is_light_gray] = 255

    return Image.fromarray(cleaned.astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Extract signal trace from cleaned image
# ─────────────────────────────────────────────────────────────────────────────

def extract_trace(image: Image.Image) -> np.ndarray:
    """
    Find the darkest continuous line in the image.
    For each column, find the row of the darkest pixel = signal amplitude.
    """
    gray   = np.array(image.convert("L"), dtype=np.float32)
    h, w   = gray.shape

    inverted = 255.0 - gray
    inverted[inverted < 50] = 0           # remove faint noise

    raw = np.zeros(w)
    for col in range(w):
        col_vals = inverted[:, col]
        if col_vals.max() > 0:
            rows    = np.arange(h, dtype=np.float32)
            weights = col_vals / col_vals.sum()
            raw[col] = np.dot(weights, rows)
        else:
            raw[col] = np.nan

    # Interpolate gaps
    nans = np.isnan(raw)
    if nans.any() and not nans.all():
        x_ok = np.where(~nans)[0]
        raw  = np.interp(np.arange(w), x_ok, raw[x_ok])

    # Flip: low row = top of image = high ECG amplitude
    raw = raw.max() - raw
    return raw


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Detect R-peaks and extract cleanest single beat
# ─────────────────────────────────────────────────────────────────────────────

def segment_best_beat(signal: np.ndarray, sampling_rate: int = 300) -> np.ndarray:
    """
    Use neurokit2 to find R-peaks and extract the most representative beat.
    Falls back to simple resampling if peak detection fails.
    """
    if signal.std() < 1e-6:
        return _resample(signal, 187)

    signal_norm = (signal - signal.mean()) / signal.std()

    try:
        cleaned  = nk.ecg_clean(signal_norm, sampling_rate=sampling_rate)
        _, info  = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        r_peaks  = info["ECG_R_Peaks"]

        if len(r_peaks) < 2:
            return _resample(signal, 187)

        half  = 93
        beats = []
        for peak in r_peaks:
            s, e = peak - half, peak + half + 1
            if s >= 0 and e <= len(signal):
                beats.append(signal[s:e])

        if not beats:
            return _resample(signal, 187)

        beats_arr = np.array(beats)
        median    = np.median(beats_arr, axis=0)
        distances = [np.linalg.norm(b - median) for b in beats]
        best      = beats[int(np.argmin(distances))]

        if best.max() > best.min():
            best = (best - best.min()) / (best.max() - best.min())
        return best.astype(np.float32)

    except Exception:
        return _resample(signal, 187)


def _resample(signal: np.ndarray, n: int) -> np.ndarray:
    x_in  = np.linspace(0, 1, len(signal))
    x_out = np.linspace(0, 1, n)
    out   = np.interp(x_out, x_in, signal)
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_signal_from_image(
    image:         Image.Image,
    sampling_rate: int = 300,
) -> tuple[np.ndarray, dict]:
    """
    Full pipeline: any ECG image → 187-point signal + debug info.

    Automatically handles full 12-lead reports and single lead strips.
    """
    debug = {"report_type": detect_report_type(image)}

    if debug["report_type"] == "full_report":
        # Locate header, lead grid, rhythm strip
        regions               = locate_ecg_regions(image)
        debug["regions"]      = regions
        debug["annotated"]    = annotate_regions(image, regions)
        rhythm_strip          = crop_rhythm_strip(image, regions)
        debug["rhythm_strip"] = rhythm_strip
    else:
        rhythm_strip          = image
        debug["rhythm_strip"] = image

    # Remove grid from the rhythm strip
    cleaned              = remove_ecg_grid(rhythm_strip)
    debug["cleaned"]     = cleaned

    # Extract trace
    raw_trace            = extract_trace(cleaned)
    debug["raw_trace"]   = raw_trace

    # Segment best beat
    signal               = segment_best_beat(raw_trace, sampling_rate)
    debug["signal"]      = signal
    debug["n_peaks"]     = _count_peaks(raw_trace, sampling_rate)

    return signal, debug


def _count_peaks(signal: np.ndarray, sampling_rate: int) -> int:
    """Count R-peaks — used as a quality check."""
    try:
        norm    = (signal - signal.mean()) / (signal.std() + 1e-6)
        cleaned = nk.ecg_clean(norm, sampling_rate=sampling_rate)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)
        return len(info["ECG_R_Peaks"])
    except Exception:
        return 0


def preprocess_for_model(
    signal: np.ndarray,
    mean:   np.ndarray,
    std:    np.ndarray,
) -> np.ndarray:
    """Apply per-feature normalisation from training data."""
    std_safe = np.where(std < 1e-6, 1.0, std)
    return (signal - mean) / std_safe
