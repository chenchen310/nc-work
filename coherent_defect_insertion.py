"""
coherent_defect_insertion.py
============================================================
Physically-motivated virtual-defect generator for partially-coherent
brightfield wafer inspection (KLA BBP-style).

核心：把 defect 當成「場（field）」的擾動，相干地加到當地背景場後再取 |.|^2，
讓合成訊號自然帶出「與 pattern 場干涉」的長相（亮點 / 暗點 / 雙極），
而不是在強度域加一個對稱 PSF blob。

Model (single-coherent-mode surrogate, "Tier 1"):

    E_total(x) = sqrt(I_bg(x)) * exp(i*phi_bg(x))  +  a_d * h_ab(x - x0)
    I_new(x)   = |E_total(x)|^2
               = I_bg                                   # 背景
               + |a_d|^2 |h|^2                          # 自身項 ~ 強度 PSF
               + 2*Re{ sqrt(I_bg) e^{-i phi_bg} a_d h } # 干涉項 <-- 讓 defect 不像 PSF

  h_ab   : 由「像差/離焦 pupil」得到的複數 amplitude PSF (ASF)
  a_d    : 複數 defect 強度 (模 = 訊號大小; 相位 = 決定亮/暗/雙極)
  phi_bg : 未知背景場相位 -> 以取樣涵蓋 (見 NOTE)

NOTE (唯一物理近似):
  我們只量到強度 I，遺失背景場相位。sqrt(I) 是「相干物體 + 實正反射率」下的
  精確場模；一般情況是好用 surrogate。未知相對相位以「取樣」涵蓋 —— 對訓練
  資料產生器反而是優點：讓合成 defect 橫跨亮<->暗，貼近真實多樣性。

只依賴 numpy。
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# 1) Zernike (Noll index) —— 用來合成離焦/像差
# ---------------------------------------------------------------------------
_NOLL = {4: (2, 0),                 # defocus (離焦)
         5: (2, -2), 6: (2, 2),     # astigmatism (像散)
         7: (3, -1), 8: (3, 1),     # coma (彗差)
         9: (3, -3), 10: (3, 3),    # trefoil (三葉)
         11: (4, 0)}                # spherical (球差)


def _zernike_radial(n, m, rho):
    m = abs(m)
    R = np.zeros_like(rho)
    for k in range((n - m) // 2 + 1):
        c = ((-1) ** k * math.factorial(n - k) /
             (math.factorial(k) *
              math.factorial((n + m) // 2 - k) *
              math.factorial((n - m) // 2 - k)))
        R = R + c * rho ** (n - 2 * k)
    return R


def _zernike(n, m, rho, theta):
    if m >= 0:
        return _zernike_radial(n, m, rho) * np.cos(m * theta)
    return _zernike_radial(n, -m, rho) * np.sin(-m * theta)


# ---------------------------------------------------------------------------
# 2) 像差 pupil -> 複數 amplitude PSF (ASF)
# ---------------------------------------------------------------------------
def aberrated_pupil(N, pixel_pitch, wavelength, NA, zernike_waves=None,
                    shift=(0.0, 0.0)):
    """回傳中心化的複數 pupil P(fx,fy) 與 oversample ratio。
    shift=(dy,dx) 以「像素」為單位，透過線性相位在 pupil 上實現 PSF 次像素平移。
    """
    f = np.fft.fftshift(np.fft.fftfreq(N, d=pixel_pitch))    # cycles / length
    FX, FY = np.meshgrid(f, f)
    fc = NA / wavelength                                     # coherent cutoff
    rho = np.sqrt(FX ** 2 + FY ** 2) / fc
    theta = np.arctan2(FY, FX)
    aperture = (rho <= 1.0)
    P = aperture.astype(np.complex128)                      # 圓孔徑, apodization=1
    if zernike_waves:
        W = np.zeros((N, N))
        rc = np.clip(rho, 0.0, 1.0)
        for noll, coeff in zernike_waves.items():
            n, m = _NOLL[noll]
            W += coeff * _zernike(n, m, rc, theta)          # W 以「waves」為單位
        P *= np.exp(1j * 2 * np.pi * W) * aperture
    dy, dx = shift                                          # 次像素平移 (Fourier shift)
    P *= np.exp(-1j * 2 * np.pi * (FX * dx * pixel_pitch + FY * dy * pixel_pitch))
    oversample = fc / (1.0 / (2.0 * pixel_pitch))           # <1 => 取樣足夠
    return P, oversample


def amplitude_psf(P):
    """從中心化 pupil 得到中心化的複數 amplitude PSF (ASF)，peak 正規化為 1。"""
    h = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(P)))
    return h / np.abs(h).max()


# ---------------------------------------------------------------------------
# 3) 局部統計 (SNR / 雜訊都要用「局部」定義才有意義)
# ---------------------------------------------------------------------------
def _patch(arr, y0, x0, half):
    y, x = int(round(y0)), int(round(x0))
    return arr[max(0, y - half):y + half + 1, max(0, x - half):x + half + 1]


def _local_std(arr, y0, x0, half=10):
    return float(_patch(arr, y0, x0, half).std()) + 1e-9


def _local_rms(arr, y0, x0, half=10):
    return float(np.sqrt(np.mean(_patch(arr, y0, x0, half) ** 2))) + 1e-12


def _scale_to_snr(E_bg, E_unit, y0, x0, sigma, target_snr, half=10):
    """對「真實(非線性) ΔI 的局部 RMS」做 bisection 求 A_d。
    避開 phase≈pi/2 時干涉項趨近 0、線性縮放會過度放大進自身項的 corner case。
    """
    target = target_snr * sigma

    def rms(A):
        d = np.abs(E_bg + A * E_unit) ** 2 - np.abs(E_bg) ** 2
        return _local_rms(d, y0, x0, half)

    lo, hi = 0.0, 1.0
    while rms(hi) < target and hi < 1e6:
        hi *= 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rms(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 4) 核心：建立 defect 場 + 相干插入
# ---------------------------------------------------------------------------
def build_defect_field(N, y0, x0, rel_phase, optics, zernike_waves):
    """回傳「單位強度」的 defect 場 a_d*h (a_d=exp(i*rel_phase))，已放到 (y0,x0)。"""
    P, oversample = aberrated_pupil(
        N, optics['pixel_pitch'], optics['wavelength'], optics['NA'],
        zernike_waves=zernike_waves, shift=(y0 - N / 2.0, x0 - N / 2.0))
    return np.exp(1j * rel_phase) * amplitude_psf(P), oversample


def insert_defect(I_bg, y0, x0, optics, zernike_waves, rel_phase,
                  target_snr, sigma_noise=None, phi_bg=None,
                  E_d=None, A_d=None):
    """把一顆 defect 相干插入 I_bg。
    E_d 若給定則直接使用 (common-mode: 同一顆貼到兩張影像)；否則依 target_snr 縮放。
    回傳 (I_new, delta, info, E_d)。
    """
    N = I_bg.shape[0]
    E_bg = np.sqrt(np.clip(I_bg, 0.0, None)).astype(np.complex128)
    if phi_bg is not None:
        E_bg = E_bg * np.exp(1j * phi_bg)

    if E_d is None:
        E_unit, oversample = build_defect_field(N, y0, x0, rel_phase,
                                                optics, zernike_waves)
        if sigma_noise is None:
            sigma_noise = _local_std(I_bg, y0, x0)
        A_d = _scale_to_snr(E_bg, E_unit, y0, x0, sigma_noise, target_snr)
        E_d = A_d * E_unit
    else:
        oversample = float('nan')

    I_new = np.abs(E_bg + E_d) ** 2                         # <-- 相干疊加後取模平方
    delta = I_new - I_bg
    info = dict(A_d=None if A_d is None else float(A_d),
                oversample=float(oversample),
                delta_center=float(delta[int(round(y0)), int(round(x0))]))
    return I_new, delta, info, E_d


# ---------------------------------------------------------------------------
# 5) 隨機化 (離焦/像差) + label map
# ---------------------------------------------------------------------------
def sample_aberration(rng, defocus=0.6, mid=0.15, high=0.08):
    """回傳 Zernike 係數 (waves)。離焦給大範圍、高階像差給小範圍。
    範圍應以 §2 估到的殘餘像差量級校準，別讓合成 PSF 比真實乾淨太多。"""
    z = {4: rng.uniform(-defocus, defocus)}
    for noll in (5, 6, 7, 8):
        z[noll] = rng.uniform(-mid, mid)
    for noll in (9, 10, 11):
        z[noll] = rng.uniform(-high, high)
    return z


def gaussian_label(N, y0, x0, sigma=2.0):
    yy, xx = np.mgrid[0:N, 0:N]
    return np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) /
                  (2 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------------------
# 6) 依「Condition Table」產生訓練樣本
#    1: none            -> No
#    2: target only     -> Defect
#    3: ref only        -> No
#    4: both(identical) -> No (common-mode)
# ---------------------------------------------------------------------------
def make_sample(target_bg, ref_bg, condition, optics, rng, defect_loc=None):
    N = target_bg.shape[0]
    if defect_loc is None:
        y0, x0 = rng.uniform(24, N - 24), rng.uniform(24, N - 24)
    else:
        y0, x0 = defect_loc
    z = sample_aberration(rng)
    rel = rng.uniform(0.0, 2 * np.pi)
    snr = rng.uniform(*optics['snr_range'])

    tgt, ref = target_bg.copy(), ref_bg.copy()
    label = np.zeros((N, N), np.float32)
    kw = dict(optics=optics, zernike_waves=z, rel_phase=rel, target_snr=snr)

    if condition == 2:                                     # target only -> Defect
        tgt, _, _, _ = insert_defect(tgt, y0, x0, **kw)
        label = gaussian_label(N, y0, x0)
    elif condition == 3:                                   # ref only -> No
        ref, _, _, _ = insert_defect(ref, y0, x0, **kw)
    elif condition == 4:                                   # both (同一顆) -> No
        tgt, _, _, E_d = insert_defect(tgt, y0, x0, **kw)
        ref, _, _, _ = insert_defect(ref, y0, x0, optics=optics, zernike_waves=z,
                                     rel_phase=rel, target_snr=snr, E_d=E_d)

    inp = np.stack([tgt, ref], axis=0).astype(np.float32)  # 2-channel U-Net input
    meta = dict(cond=condition, y0=y0, x0=x0, snr=snr, rel_phase=rel, z=z)
    return inp, label, meta


def make_nuisance_negative(target_bg, ref_bg, optics, rng, defect_loc=None):
    """§4② 關鍵補丁：target 與 ref 貼「不同」擾動 (相位/強度/次像素位置皆異)，
    模擬 process variation 的非 common-mode 殘差；label 一律 No。
    這才是逼近真實 nuisance 的困難負例。"""
    N = target_bg.shape[0]
    if defect_loc is None:
        y0, x0 = rng.uniform(24, N - 24), rng.uniform(24, N - 24)
    else:
        y0, x0 = defect_loc

    def jitter():
        return y0 + rng.uniform(-1.5, 1.5), x0 + rng.uniform(-1.5, 1.5)

    yt, xt = jitter()
    yr, xr = jitter()
    tgt, _, _, _ = insert_defect(target_bg.copy(), yt, xt, optics=optics,
                                 zernike_waves=sample_aberration(rng),
                                 rel_phase=rng.uniform(0, 2 * np.pi),
                                 target_snr=rng.uniform(*optics['snr_range']))
    ref, _, _, _ = insert_defect(ref_bg.copy(), yr, xr, optics=optics,
                                 zernike_waves=sample_aberration(rng),
                                 rel_phase=rng.uniform(0, 2 * np.pi),
                                 target_snr=rng.uniform(*optics['snr_range']))
    inp = np.stack([tgt, ref], axis=0).astype(np.float32)
    label = np.zeros((N, N), np.float32)                   # No，即使 difference 非零
    return inp, label, dict(cond='nuisance', y0=y0, x0=x0)


# ===========================================================================
# 自我測試：驗證「同一顆 defect、不同相位 -> 亮/暗/雙極」的核心物理
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 256
    optics = dict(wavelength=266.0, NA=0.90, pixel_pitch=30.0, snr_range=(0.4, 1.2))

    yy, xx = np.mgrid[0:N, 0:N]
    base = 1.0 + 0.6 * np.cos(2 * np.pi * xx / 12.0)
    target_bg = base + 0.02 * rng.standard_normal((N, N))
    ref_bg = np.roll(base, 1, axis=1) * 1.01 + 0.02 * rng.standard_normal((N, N))

    res = optics['wavelength'] / (2 * optics['NA'])
    _, ovs = aberrated_pupil(N, optics['pixel_pitch'], optics['wavelength'], optics['NA'])
    print("=== optics sanity ===")
    print(f"optical resolution lambda/2NA = {res:6.1f} nm  (pixel = {optics['pixel_pitch']:.0f} nm)")
    print(f"PSF FWHM ~ {res/optics['pixel_pitch']:.1f} px | oversample = {ovs:.3f} (<1 OK)")
    assert ovs < 1.0

    y0, x0 = 128.0, 120.0
    z_small = {4: 0.10, 5: 0.03, 6: -0.02, 7: 0.02, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.03}
    print("\n=== same defect, sweep relative phase (SNR fixed=1.0) ===")
    centers = {}
    for name, ph in [("phase=0  ", 0.0), ("phase=pi/2", np.pi / 2), ("phase=pi ", np.pi)]:
        _, delta, info, _ = insert_defect(target_bg, y0, x0, optics=optics,
                                          zernike_waves=z_small, rel_phase=ph, target_snr=1.0)
        centers[name] = info['delta_center']
        print(f"{name}: delta@center={info['delta_center']:+.4f} | A_d={info['A_d']:.4f}")
    assert centers["phase=0  "] > 0 > centers["phase=pi "]
    print("OK -> phase=0 亮點, phase=pi 暗點 (干涉項變號)")

    print("\n=== condition table sanity ===")
    for cond in (1, 2, 3, 4):
        inp, label, _ = make_sample(target_bg, ref_bg, cond, optics, rng, (90.0, 140.0))
        print(f"cond {cond}: label_sum={label.sum():8.2f} | "
              f"Δtarget={np.abs(inp[0]-target_bg).sum():8.2f} Δref={np.abs(inp[1]-ref_bg).sum():8.2f}")
    print("\nALL CHECKS PASSED")
