"""
visualize_defects.py
============================================================
把 coherent_defect_insertion 產生的合成 defect 畫出來，直觀驗證兩件事：

  fig1_interference_physics.png
    (上排) 同一顆 defect、同一位置與像差，只改「與背景場的相對相位」
           -> 亮點 / 雙極 / 暗點   (§1 干涉項在作用)
    (下排) 同一顆 defect (同相位/像差/SNR)，放到 亮處 / 邊緣 / 暗處
           -> signature 隨當地 pattern 而變 (這就是「defect 為何不像 PSF」)

  fig2_condition_table.png
    Condition Table (cond2 Defect / cond3 ref-only / cond4 both) + 困難 nuisance 負例，
    每列顯示 target / reference / difference(target-ref) / label，對齊你的 U-Net pipeline。

用法:  python3 visualize_defects.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless 存檔
import matplotlib.pyplot as plt

import coherent_defect_insertion as cdi

OPTICS = dict(wavelength=266.0, NA=0.90, pixel_pitch=30.0, snr_range=(0.4, 1.2))
N = 256


def make_backgrounds(rng):
    """垂直 line-space pattern；reference 帶輕微 shift+gain (模擬 process variation)。"""
    _, xx = np.mgrid[0:N, 0:N]
    base = 1.0 + 0.6 * np.cos(2 * np.pi * xx / 12.0)          # 正值強度, period=12px
    target = base + 0.02 * rng.standard_normal((N, N))
    ref = np.roll(base, 1, axis=1) * 1.01 + 0.02 * rng.standard_normal((N, N))
    return target.astype(np.float64), ref.astype(np.float64)


def crop(a, y0, x0, half):
    y, x = int(round(y0)), int(round(x0))
    return a[y - half:y + half, x - half:x + half]


def _imshow(ax, img, title, cmap="gray", vlim=None, diverging=False):
    if diverging:
        v = np.abs(img).max() + 1e-9
        im = ax.imshow(img, cmap="seismic", vmin=-v, vmax=v)
    elif vlim is not None:
        im = ax.imshow(img, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
    else:
        im = ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


def radial_profile(img, cy, cx, rmax):
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    return np.array([img[r == k].mean() if np.any(r == k) else 0.0 for k in range(rmax)])


def figure_broadband(target_bg):
    half = 24
    z = {4: 0.05}
    band = cdi.make_band(center=270.0, fwhm=90.0, n=9)     # 換成你的實際 S(λ)
    y0, x0 = 128.0, 120.0
    c = N // 2
    # 強度 PSF：mono vs broadband(Σ w|h_λ|²)
    P, _ = cdi.aberrated_pupil(N, OPTICS['pixel_pitch'], OPTICS['wavelength'], OPTICS['NA'], z)
    psf_mono = np.abs(cdi.amplitude_psf(P)) ** 2
    psf_bb = np.zeros((N, N))
    ws = np.array([w for _, w in band]); ws /= ws.sum()
    for (lam, _), w in zip(band, ws):
        Pl, _ = cdi.aberrated_pupil(N, OPTICS['pixel_pitch'], lam, OPTICS['NA'], z)
        psf_bb += w * np.abs(cdi.amplitude_psf(Pl)) ** 2
    # defect ΔI：mono vs broadband (同背景/位置/相位/SNR)
    _, d_mono, _, _ = cdi.insert_defect(target_bg, y0, x0, optics=OPTICS,
                                        zernike_waves=z, rel_phase=0.0, target_snr=1.0)
    _, d_bb, _ = cdi.insert_defect_broadband(target_bg, y0, x0, OPTICS, z, 0.0, 1.0, band)
    fig, ax = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    _imshow(ax[0, 0], np.log10(crop(psf_mono, c, c, half) + 1e-6), "mono-λ intensity PSF (log)", cmap="viridis")
    _imshow(ax[0, 1], np.log10(crop(psf_bb, c, c, half) + 1e-6), "broadband intensity PSF (log)", cmap="viridis")
    pm, pb = radial_profile(psf_mono, c, c, half), radial_profile(psf_bb, c, c, half)
    ax[0, 2].semilogy(pm / pm.max(), label="mono-λ")
    ax[0, 2].semilogy(pb / pb.max(), label="broadband")
    ax[0, 2].set_title("PSF radial profile (rings damped)")
    ax[0, 2].set_xlabel("radius [px]"); ax[0, 2].legend(); ax[0, 2].grid(True, which="both", alpha=0.3)
    _imshow(ax[1, 0], crop(d_mono, y0, x0, half), "mono-λ  defect ΔI", diverging=True)
    _imshow(ax[1, 1], crop(d_bb, y0, x0, half), "broadband  defect ΔI", diverging=True)
    qm, qb = radial_profile(np.abs(d_mono), y0, x0, half), radial_profile(np.abs(d_bb), y0, x0, half)
    ax[1, 2].semilogy(qm / qm.max(), label="mono-λ")
    ax[1, 2].semilogy(qb / qb.max(), label="broadband")
    ax[1, 2].set_title("|ΔI| radial profile")
    ax[1, 2].set_xlabel("radius [px]"); ax[1, 2].legend(); ax[1, 2].grid(True, which="both", alpha=0.3)
    fig.suptitle("Fig.3  Mono-λ vs Broadband   |   broadband = incoherent sum over λ "
                 "-> outer rings wash out (surviving rings ~ λ/Δλ)", fontsize=10)
    out = "fig3_broadband_vs_mono.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 1 : 干涉物理
# ---------------------------------------------------------------------------
def figure_interference(target_bg):
    half = 24
    top_z = {4: 0.10, 6: 0.06, 7: 0.20, 11: 0.05}      # 帶 coma -> π/2 呈明顯雙極
    bot_z = {4: 0.05}                                  # 近繞射極限, 讓變化只歸因於 pattern
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.6), constrained_layout=True)

    # 上排：相位掃描 (同位置、同像差、SNR 固定)
    y0, x0 = 128.0, 120.0
    for j, (name, ph) in enumerate([("relative phase = 0\n(bright)", 0.0),
                                    ("relative phase = π/2\n(bipolar)", np.pi / 2),
                                    ("relative phase = π\n(dark)", np.pi)]):
        _, delta, info, _ = cdi.insert_defect(target_bg, y0, x0, optics=OPTICS,
                                              zernike_waves=top_z, rel_phase=ph,
                                              target_snr=1.0)
        _imshow(axes[0, j], crop(delta, y0, x0, half),
                f"{name}\nΔI@center={info['delta_center']:+.2f}", diverging=True)

    # 下排：pattern 依存 (同顆 defect 放到 亮/邊/暗)
    locs = [("bright peak", 128.0, 120.0),
            ("edge", 128.0, 123.0),
            ("dark trough", 128.0, 126.0)]
    for j, (name, y0, x0) in enumerate(locs):
        _, delta, info, _ = cdi.insert_defect(target_bg, y0, x0, optics=OPTICS,
                                              zernike_waves=bot_z, rel_phase=0.0,
                                              target_snr=1.0)
        _imshow(axes[1, j], crop(delta, y0, x0, half),
                f"same defect @ {name}\nΔI@center={info['delta_center']:+.2f}", diverging=True)

    fig.suptitle("Fig.1  Difference image  ΔI  of one defect     |     "
                 "Top: sweep relative phase (bright / bipolar / dark)     |     "
                 "Bottom: same defect on bright / edge / dark background",
                 fontsize=10)
    out = "fig1_interference_physics.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2 : Condition Table + nuisance 負例
# ---------------------------------------------------------------------------
def figure_condition_table(target_bg, ref_bg):
    half = 36
    loc = (128.0, 120.0)
    rows = []

    # cond 2/3/4 用同一顆 defect (固定 seed) 以便直接對照
    for cond, tag in [(2, "cond2: target-only -> Defect"),
                      (3, "cond3: ref-only -> No"),
                      (4, "cond4: both (common-mode) -> No")]:
        rng = np.random.default_rng(42)                      # 同一顆 defect
        inp, label, _ = cdi.make_sample(target_bg, ref_bg, cond, OPTICS, rng, defect_loc=loc)
        rows.append((tag, inp[0], inp[1], label))

    # 困難 nuisance 負例 (§4②)：target/ref 貼不同擾動、label=No
    rng = np.random.default_rng(7)
    inp_n, label_n, _ = cdi.make_nuisance_negative(target_bg, ref_bg, OPTICS, rng, defect_loc=loc)
    rows.append(("nuisance: diff target/ref -> No", inp_n[0], inp_n[1], label_n))

    ivmin = min(target_bg.min(), ref_bg.min())
    ivmax = max(target_bg.max(), ref_bg.max())
    fig, axes = plt.subplots(len(rows), 4, figsize=(13, 3.1 * len(rows)),
                             constrained_layout=True)
    col_titles = ["target", "reference", "difference (target − ref)", "label (GT)"]
    y0, x0 = loc
    for i, (tag, tgt, ref, lab) in enumerate(rows):
        _imshow(axes[i, 0], crop(tgt, y0, x0, half), col_titles[0] if i == 0 else "",
                cmap="gray", vlim=(ivmin, ivmax))
        _imshow(axes[i, 1], crop(ref, y0, x0, half), col_titles[1] if i == 0 else "",
                cmap="gray", vlim=(ivmin, ivmax))
        _imshow(axes[i, 2], crop(tgt - ref, y0, x0, half), col_titles[2] if i == 0 else "",
                diverging=True)
        _imshow(axes[i, 3], crop(lab, y0, x0, half), col_titles[3] if i == 0 else "",
                cmap="magma", vlim=(0, 1))
        axes[i, 0].set_ylabel(tag, fontsize=10)
    fig.suptitle("Fig.2  Condition table + hard nuisance negative (U-Net pipeline view)     |     "
                 "cond4 common-mode cancels in difference;  nuisance difference != 0 but label = No",
                 fontsize=10)
    out = "fig2_condition_table.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    target_bg, ref_bg = make_backgrounds(rng)

    res = OPTICS['wavelength'] / (2 * OPTICS['NA'])
    print(f"optical resolution λ/2NA = {res:.1f} nm | pixel = {OPTICS['pixel_pitch']:.0f} nm "
          f"| PSF FWHM ~ {res/OPTICS['pixel_pitch']:.1f} px")

    f1 = figure_interference(target_bg)
    f2 = figure_condition_table(target_bg, ref_bg)
    f3 = figure_broadband(target_bg)
    print("saved:", f1)
    print("saved:", f2)
    print("saved:", f3)
