"""
visualize_matched_filter.py
============================================================
畫出 §3 的 whitened matched filter：在有色 wafer noise 下，白化如何把
SNR<1 的 defect 從結構化背景中拉出來，並勝過未白化的 matched filter。
輸出 fig4_whitened_matched_filter.png

用法:  python3 visualize_matched_filter.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import coherent_defect_insertion as cdi
import whitened_matched_filter as wmf
from visualize_defects import _imshow, crop, OPTICS, N


def _mark(ax, loc, color="lime"):
    y, x = loc
    ax.add_patch(Circle((x, y), 8, fill=False, ec=color, lw=1.6))


def figure_matched_filter():
    _, xx = np.mgrid[0:N, 0:N]
    base = 1.0 + 0.6 * np.cos(2 * np.pi * xx / 12.0)            # 強 pattern -> 有色雜訊

    def make_pair(seed):
        g = np.random.default_rng(seed)
        tgt = base + 0.02 * g.standard_normal((N, N))
        ref = np.roll(base, 1, axis=1) * 1.01 + 0.02 * g.standard_normal((N, N))
        return tgt, ref

    # 40 張無缺陷 difference -> 估 noise PSD
    diffs = np.array([make_pair(1000 + s)[0] - make_pair(1000 + s)[1] for s in range(40)])
    Pn = wmf.estimate_noise_psd(diffs)

    # template (代表 PSF) 與真實 defect 像差「不同」(realistic mismatch)
    z_tmpl = {4: 0.10, 7: 0.06, 6: 0.04}
    z_defect = {4: 0.16, 7: 0.11, 6: -0.03}
    templates = wmf.build_psf_templates(N, OPTICS, z_tmpl)
    loc = (128.0, 120.0)
    tgt, ref = make_pair(7)
    d_noise_std = float((tgt - ref).std())
    tgt_def, delta, info, _ = cdi.insert_defect(
        tgt, loc[0], loc[1], optics=OPTICS, zernike_waves=z_defect,
        rel_phase=0.7, target_snr=0.8, sigma_noise=d_noise_std)
    d = tgt_def - ref

    out_white = wmf.whitened_matched_filter(d, templates, np.ones_like(Pn))
    out_wht = wmf.whitened_matched_filter(d, templates, Pn)
    det_w = wmf.detection_snr(out_white["score"], loc)
    det_h = wmf.detection_snr(out_wht["score"], loc)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8.6), constrained_layout=True)

    # ---- 上排：材料 ----
    _imshow(ax[0, 0], diffs[0], "defect-free difference\n(colored wafer noise)", diverging=True)
    psd = np.log10(np.fft.fftshift(Pn) + 1e-9)
    cap = np.percentile(psd, 99.5); psd = np.minimum(psd, cap)  # 壓掉人工放大的 DC
    _imshow(ax[0, 1], psd, "estimated noise PSD  log10 (colored)", cmap="viridis")
    _imshow(ax[0, 2], crop(templates[0], N // 2, N // 2, 20), "PSF template  Re(h)", diverging=True)

    # ---- 下排：raw / white MF / whitened MF ----
    _imshow(ax[1, 0], d, "difference + sub-noise defect\n(raw SNR ~ 0.8)", diverging=True)
    _mark(ax[1, 0], loc)
    _imshow(ax[1, 1], out_white["score"],
            f"white MF score  (peak/bg = {det_w['snr']:.1f})", cmap="inferno")
    _mark(ax[1, 1], loc)
    _imshow(ax[1, 2], out_wht["score"],
            f"whitened MF score  (peak/bg = {det_h['snr']:.1f})", cmap="inferno")
    _mark(ax[1, 2], loc)

    fig.suptitle("Fig.4  Whitened matched filter (§3)   |   colored wafer noise -> whitening (1/P_n) "
                 "pulls the sub-noise (SNR<1) defect out;  matched-subspace T = z_re^2 + z_im^2 "
                 "handles bright/dark/bipolar", fontsize=10)
    out = "fig4_whitened_matched_filter.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    print("saved:", figure_matched_filter())
