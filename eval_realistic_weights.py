"""
Phase 2 evaluation: compare the original Ballet weights against the realistic-noise
retrain on a ground-truth synthetic SNR sweep (same model as centroid_comparison.ipynb
section 17). Purely synthetic -- no image, no network.

Usage:
    python eval_realistic_weights.py /path/to/new_weights.npz
"""
import sys
import warnings
import numpy as np
from astropy.modeling.models import Moffat2D, Const2D
from astropy.modeling.fitting import TRFLSQFitter
from astropy.nddata import Cutout2D

from eloy.ballet.model import Ballet, download_weights
from eloy.centroid import ballet_centroid

FRAME, BOX = 41, 15
BETA, FWHM = 3.0, 4.0
GAMMA = FWHM / (2.0 * np.sqrt(2.0 ** (1.0 / BETA) - 1.0))
SKY, READ_NOISE = 100.0, 5.0
NOISE_FLOOR = np.sqrt(SKY + READ_NOISE**2)
SNR_GRID = np.array([2, 3, 5, 8, 12, 20, 40, 80, 160], dtype=float)
N_PER = 80
BAYER_GAINS = (1.00, 1.35, 1.35, 0.80)
_yy, _xx = np.mgrid[:FRAME, :FRAME]


def bayer_balance_image(image):
    from bandaid import bayer_balance_image as _b
    _b(image)


def make_star(rng, amp, cx, cy, bayer=False):
    clean = Moffat2D(amplitude=amp, x_0=cx, y_0=cy, gamma=GAMMA, alpha=BETA)(_xx, _yy) + SKY
    if bayer:
        clean = clean.copy()
        clean[0::2, 0::2] *= BAYER_GAINS[0]; clean[0::2, 1::2] *= BAYER_GAINS[1]
        clean[1::2, 0::2] *= BAYER_GAINS[2]; clean[1::2, 1::2] *= BAYER_GAINS[3]
    noisy = (rng.poisson(np.clip(clean, 0, None)).astype(float)
             + rng.normal(0.0, READ_NOISE, clean.shape))
    if bayer:
        bayer_balance_image(noisy)
    return noisy


def moffat_recover(data, sx, sy):
    cut = Cutout2D(data, (sx, sy), (BOX, BOX), mode="partial", fill_value=np.nan)
    arr = np.asarray(cut.data, float)
    if not np.isfinite(arr).all():
        return np.nan, np.nan
    yy, xx = np.mgrid[: arr.shape[0], : arr.shape[1]]
    bkg = float(np.nanmedian(arr)); amp = float(np.nanmax(arr) - bkg)
    g = FWHM / (2.0 * np.sqrt(2.0 ** (1.0 / 3.0) - 1.0))
    m = Moffat2D(amplitude=amp, x_0=arr.shape[1] / 2, y_0=arr.shape[0] / 2, gamma=g, alpha=3.0)
    m.x_0.bounds = (0, arr.shape[1] - 1); m.y_0.bounds = (0, arr.shape[0] - 1)
    m.gamma.bounds = (0.1, 3 * FWHM); m.alpha.bounds = (0.5, 10); m.amplitude.bounds = (0, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            fit = TRFLSQFitter()(m + Const2D(amplitude=bkg), xx, yy, arr)
        except Exception:
            return np.nan, np.nan
    return cut.to_original_position((float(fit[0].x_0.value), float(fit[0].y_0.value)))


def sweep(models, bayer=False, seed=42):
    rng = np.random.default_rng(seed)
    out = {name: {s: [] for s in SNR_GRID} for name in list(models) + ["Moffat"]}
    for snr in SNR_GRID:
        amp = snr * NOISE_FLOOR
        for _ in range(N_PER):
            cx = FRAME / 2 + rng.uniform(-1.5, 1.5)
            cy = FRAME / 2 + rng.uniform(-1.5, 1.5)
            noisy = make_star(rng, amp, cx, cy, bayer=bayer)
            sx, sy = round(cx), round(cy)
            for name, cnn in models.items():
                bx, by = ballet_centroid(noisy, np.array([[float(sx), float(sy)]]), cnn, nans=True)[0]
                e = np.hypot(bx - cx, by - cy)
                if np.isfinite(e) and e <= BOX / 2:
                    out[name][snr].append(e)
            mx, my = moffat_recover(noisy, sx, sy)
            e = np.hypot(mx - cx, my - cy)
            if np.isfinite(e) and e <= BOX / 2:
                out["Moffat"][snr].append(e)
    return out


def show(title, res):
    print(f"\n[{title}] median recovery error (px) by SNR:")
    print("  SNR   " + "  ".join(f"{int(s):>6d}" for s in SNR_GRID))
    for name, per in res.items():
        row = "  ".join((f"{np.median(per[s]):6.3f}" if per[s] else "    --") for s in SNR_GRID)
        print(f"  {name:9s} {row}")


if __name__ == "__main__":
    # Usage: python eval_realistic_weights.py [label=]path ...
    # The HuggingFace default weights are always included as "old".
    import os

    models = {"old": Ballet(download_weights())}
    for arg in sys.argv[1:]:
        if "=" in arg:
            label, path = arg.split("=", 1)
        else:
            label, path = os.path.splitext(os.path.basename(arg))[0], arg
        models[label] = Ballet(path)
    for bayer in (False, True):
        show("bayer" if bayer else "plain", sweep(models, bayer=bayer))
