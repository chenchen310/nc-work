"""
whitened_matched_filter.py
============================================================
§3 的 whitened matched filter：在「有色 (colored) wafer noise」中偵測已知形狀
的 defect。對 SNR<1、訊號形狀已知的情況，這是理論最佳的線性偵測前端
(Kay, Fundamentals of Statistical Signal Processing Vol.II, Detection Theory;
 Scharf & Friedlander, matched subspace detectors)。

物理連結 (承 coherent_defect_insertion 的前向模型)：
  difference image d 的 defect 線性主項 ≈ 2·sqrt(I_bg)·Re{a_d·h}
      = 2·sqrt(I_bg)·[Re(a_d)·Re(h) − Im(a_d)·Im(h)]
  也就是說 (把 sqrt(I_bg) 在 PSF footprint 內近似為定值) defect 訊號落在
  span{Re(h), Im(h)} 這個 2D 子空間，係數由未知複數 a_d 決定 -> 亮/暗/雙極。

偵測策略：
  1. 白化 (whitening)：wafer noise 是有色的 (pattern/gain/對位殘差集中在低頻與
     pattern 頻率)。頻域用 1/P_n(f) 加權，自動壓低「雜訊強的頻率」、放大
     「compact PSF 有能量的中高頻」。
  2. Matched filter：與 (白化後的) template 相關 -> 對已知形狀達最大輸出 SNR。
  3. 未知相位 -> matched-subspace：T = z_re² + z_im² (H0 下 ~ χ²_2)，對亮/暗/
     雙極皆敏感；不必知道 a_d 的相位。

P_n(f) 由「無缺陷 difference images」(die-to-die pairs) 的平均週期圖估計。
只依賴 numpy；template 來自 coherent_defect_insertion 的 amplitude PSF。
"""
import numpy as np
import coherent_defect_insertion as cdi


# ---------------------------------------------------------------------------
# 1) 從無缺陷 difference images 估 noise PSD  P_n(f) = E|FFT(d)|²
# ---------------------------------------------------------------------------
def estimate_noise_psd(diffs, floor_frac=1e-3, smooth=3, exclude_dc=True):
    """diffs: (M,H,W) 無缺陷 difference images。回傳 P_n (H,W)。
    平均週期圖 + box 平滑 + flooring；DC 設極大以忽略全域 gain 偏移。"""
    diffs = np.asarray(diffs, float)
    if diffs.ndim == 2:
        diffs = diffs[None]
    diffs = diffs - diffs.mean(axis=(-2, -1), keepdims=True)     # 去每張均值
    P = np.mean(np.abs(np.fft.fft2(diffs, axes=(-2, -1))) ** 2, axis=0)
    if smooth > 1:
        P = _box_smooth(P, smooth)
    P = np.maximum(P, floor_frac * P.mean())                    # 避免 1/P_n 爆掉
    if exclude_dc:
        P[0, 0] = P.max() * 1e6                                  # 忽略 DC
    return P


def _box_smooth(P, k):
    """在 fftshift 後的頻域做 k×k 箱型平滑 (穩定 PSD 估計)。"""
    ps = np.fft.fftshift(P)
    pad = k // 2
    psp = np.pad(ps, pad, mode="edge")
    out = np.zeros_like(ps)
    for i in range(k):
        for j in range(k):
            out += psp[i:i + ps.shape[0], j:j + ps.shape[1]]
    return np.fft.ifftshift(out / (k * k))


# ---------------------------------------------------------------------------
# 2) 從 amplitude PSF 建 matched-subspace template  {Re(h), Im(h)}
# ---------------------------------------------------------------------------
def build_psf_templates(N, optics, zernike_waves=None):
    """回傳 (s_re, s_im) = amplitude PSF 的實/虛部 (中心化)。
    difference-image 的線性 defect 訊號落在 span{s_re, s_im}。
    實務上請用 §2 量到的 PSF/pupil；zernike 只作殘餘像差。"""
    z = zernike_waves or {}
    P, _ = cdi.aberrated_pupil(N, optics["pixel_pitch"], optics["wavelength"],
                               optics["NA"], z)
    h = cdi.amplitude_psf(P)                                    # 複數, 中心化, peak=1
    return np.real(h).copy(), np.imag(h).copy()


# ---------------------------------------------------------------------------
# 3) whitened matched filter (+ 未知相位的 matched-subspace 統計)
# ---------------------------------------------------------------------------
def _mf_response(X, s, Pn):
    """單一 template 的 whitened matched filter 原始響應 r = IFFT{conj(S)·X/Pn}。
    S 由「原點對齊」的 template 得到，使 r 的 peak 落在 defect 位置。"""
    S = np.fft.fft2(np.fft.ifftshift(s))                        # 中心化 -> 原點對齊
    r = np.fft.ifft2(np.conj(S) * X / Pn).real                 # 白化匹配濾波 (相關)
    Ew = float(np.sum(np.abs(S) ** 2 / Pn))                    # 白化後 template 能量
    return r, Ew


def _robust_z(a):
    """用 median/MAD 把響應正規化成單位標準差的 z-score (defect 只占少數像素)。"""
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-12
    return (a - med) / (1.4826 * mad)


def whitened_matched_filter(x, templates, Pn, robust_norm=True):
    """x: (H,W) difference image。templates: (s_re, s_im)。Pn: noise PSD。
    回傳 dict:
      z_re, z_im : 兩個 quadrature template 的 z-normalized 響應
      T          : matched-subspace 統計 = z_re² + z_im²  (H0 下 ~ χ²_2, 或 χ²_1)
      score      : sqrt(T)，可直接當偵測圖 / U-Net 的額外輸入通道
    """
    x = np.asarray(x, float)
    X = np.fft.fft2(x - x.mean())                              # 去 DC
    s_re, s_im = templates

    r_re, e_re = _mf_response(X, s_re, Pn)
    z_re = _robust_z(r_re) if robust_norm else r_re * (x.size / np.sqrt(e_re))

    if np.sum(s_im ** 2) > 1e-6 * np.sum(s_re ** 2):          # 有像差 -> Im(h) 有效
        r_im, e_im = _mf_response(X, s_im, Pn)
        z_im = _robust_z(r_im) if robust_norm else r_im * (x.size / np.sqrt(e_im))
    else:                                                     # 無像差 -> 退化為單 template
        z_im = np.zeros_like(z_re)

    T = z_re ** 2 + z_im ** 2
    return dict(z_re=z_re, z_im=z_im, T=T, score=np.sqrt(T))


# ---------------------------------------------------------------------------
# 4) 偵測指標：peak (真值鄰域最大) vs 背景 robust std
# ---------------------------------------------------------------------------
def detection_snr(zmap, loc, tol=2, exclude=8):
    y, x = int(round(loc[0])), int(round(loc[1]))
    peak = float(zmap[max(0, y - tol):y + tol + 1, max(0, x - tol):x + tol + 1].max())
    mask = np.ones_like(zmap, bool)
    mask[max(0, y - exclude):y + exclude + 1, max(0, x - exclude):x + exclude + 1] = False
    bg = zmap[mask]
    bg_std = float(np.median(np.abs(bg - np.median(bg))) * 1.4826) + 1e-9
    ay, ax = np.unravel_index(np.argmax(zmap), zmap.shape)
    localized = (abs(ay - y) <= 3 and abs(ax - x) <= 3)
    return dict(peak=peak, bg_std=bg_std, snr=peak / bg_std,
                argmax=(int(ay), int(ax)), localized=bool(localized))


# ===========================================================================
# 自我測試：sub-noise defect (raw SNR<1) 被 whitened MF 拉出來；且勝過 white MF
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 256
    optics = dict(wavelength=266.0, NA=0.90, pixel_pitch=30.0, snr_range=(0.4, 1.2))
    _, xx = np.mgrid[0:N, 0:N]
    base = 1.0 + 0.6 * np.cos(2 * np.pi * xx / 12.0)            # 強 pattern (有色 noise 來源)

    def make_pair(seed):
        g = np.random.default_rng(seed)
        tgt = base + 0.02 * g.standard_normal((N, N))
        ref = np.roll(base, 1, axis=1) * 1.01 + 0.02 * g.standard_normal((N, N))
        return tgt, ref

    # ---- 估 P_n：40 張無缺陷 difference ----
    diffs = []
    for s in range(40):
        t, r = make_pair(1000 + s)
        diffs.append(t - r)
    diffs = np.array(diffs)
    Pn = estimate_noise_psd(diffs)
    _pn = Pn.copy(); _pn[0, 0] = np.median(Pn)                   # 排除人工放大的 DC 再看有色度
    color = float(np.percentile(_pn, 99.9) / np.median(_pn))
    print("=== noise PSD (colored) ===")
    print(f"P_n 有色度 (99.9pct/median, 去DC) = {color:.1f}  (>>1 => 有色雜訊，值得白化)")

    # ---- 種一顆 sub-noise defect：讓它在 difference 上 SNR<1 ----
    #      注意 template 與真實 defect 的像差「不同」(realistic mismatch)。
    z_tmpl = {4: 0.10, 7: 0.06, 6: 0.04}                        # 模型/量到的代表 PSF
    z_defect = {4: 0.16, 7: 0.11, 6: -0.03}                     # 真實 defect 像差 (與 template 不同)
    templates = build_psf_templates(N, optics, z_tmpl)
    loc = (128.0, 120.0)
    tgt, ref = make_pair(7)
    d_noise_std = float((tgt - ref).std())
    tgt_def, delta, info, _ = cdi.insert_defect(
        tgt, loc[0], loc[1], optics=optics, zernike_waves=z_defect,
        rel_phase=0.7, target_snr=0.8, sigma_noise=d_noise_std)  # 用 diff 噪聲當 SNR 基準
    d = tgt_def - ref                                           # 帶缺陷的 difference

    # raw difference 的偵測 SNR (defect 幾乎淹沒在 pattern 殘差裡)
    sig_rms = float(np.sqrt((delta[118:139, 110:131] ** 2).mean()))   # ±10px 局部訊號 rms
    raw = detection_snr(np.abs(d - d.mean()), loc)
    print("\n=== raw difference (no filter) ===")
    print(f"defect signal rms/diff-noise ≈ {sig_rms/d_noise_std:.2f}  (<1, sub-noise)")
    print(f"raw |d| detection: peak/bg={raw['snr']:.2f}  localized={raw['localized']} "
          f"argmax={raw['argmax']} (背景滿是結構化 stripe 殘差 -> 易誤報)")

    # ---- white MF (Pn=const) vs whitened MF (Pn 估計) ----
    out_white = whitened_matched_filter(d, templates, np.ones_like(Pn))
    out_white_det = detection_snr(out_white["score"], loc)
    out_wht = whitened_matched_filter(d, templates, Pn)
    out_wht_det = detection_snr(out_wht["score"], loc)

    print("\n=== matched filter (score = sqrt(z_re²+z_im²)) ===")
    print(f"white   MF: peak/bg={out_white_det['snr']:5.2f}  localized={out_white_det['localized']} "
          f"argmax={out_white_det['argmax']}")
    print(f"whitened MF: peak/bg={out_wht_det['snr']:5.2f}  localized={out_wht_det['localized']} "
          f"argmax={out_wht_det['argmax']}")

    # ---- 斷言 ----
    assert color > 5, "noise 應為有色 (否則白化無意義)"
    assert out_wht_det["localized"], "whitened MF 應正確定位 defect"
    assert out_wht_det["snr"] > out_white_det["snr"], "白化應勝過 white MF (有色雜訊下)"
    assert out_wht_det["snr"] > raw["snr"] + 2, "whitened MF 應把 sub-noise defect 拉到遠高於 raw"
    assert out_wht_det["snr"] > 5.0, "whitened MF 偵測 SNR 應可越過門檻"
    print("\nOK -> 白化匹配濾波把 SNR<1 的 defect 拉到 "
          f"{out_wht_det['snr']:.1f}σ，且勝過 white MF ({out_white_det['snr']:.1f}σ)")
    print("\nALL CHECKS PASSED")
