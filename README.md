# Coherent virtual-defect insertion (partially-coherent brightfield inspection)

針對 KLA BBP 光學晶圓缺陷檢測的 **物理正確 virtual-defect 產生器**。
把 defect 當成「場 (field)」的擾動、相干加到當地背景場後再取 `|·|²`，
讓合成訊號自然帶出「與 pattern 場干涉」的長相（亮 / 暗 / 雙極），
取代「在強度域加一個對稱 PSF blob」的做法（那會造成 train/test domain gap）。

> 定調：sub-resolution defect 的偵測瓶頸不是解析度，而是 **SNR / contrast**；
> difference image 上的 defect 訊號由 defect 場與 pattern 場的**干涉**主導，
> 所以「不像乾淨 PSF」是正常物理。
> 參考 Zhu et al., *Optical wafer defect inspection at the 10 nm technology node and beyond*,
> Int. J. Extreme Manuf. (2022), DOI:10.1088/2631-7990/ac64d7.

## 成像模型 (Tier 1: single-coherent-mode surrogate)

```
E_total(x) = sqrt(I_bg(x)) · exp(i·φ_bg(x))  +  a_d · h_ab(x − x0)
I_new(x)   = |E_total(x)|²
           = I_bg  +  |a_d|²·|h|²  +  2·Re{ sqrt(I_bg)·e^(−iφ_bg)·a_d·h }
             背景      自身項(~強度PSF)   干涉項  ← 讓 defect 隨 pattern 而變
```

- `h_ab` = 由「像差/離焦 pupil」得到的複數 amplitude PSF：
  `P(f) = A(ρ)·exp(i·2π·Σ c_j Z_j(ρ,θ))`，`ρ = |f|/(NA/λ)`，`h_ab = F⁻¹{P}`。
  `Z_j` 為 Zernike（Noll 4=defocus, 5/6=astig, 7/8=coma, 11=spherical），`c_j` 單位為 waves。
- `a_d = A_d·e^(iψ_d)`：模 = 訊號強度、相位 = 決定亮/暗/雙極。
- `φ_bg`（未知背景相位）以**取樣**涵蓋——對訓練資料產生器是優點，自動橫跨亮↔暗。
- 離焦距離換算：`c_defocus[waves] ≈ NA²·Δz / (2λ)`。
- 取樣條件：`Δx ≤ λ/(2·NA)`（程式內以 `oversample<1` assert）。

## 檔案

| 檔案 | 內容 |
|------|------|
| `coherent_defect_insertion.py` | 產生器函式庫：pupil/Zernike、相干插入 `insert_defect`、依 Condition Table 的 `make_sample`、困難負例 `make_nuisance_negative`。`python3 coherent_defect_insertion.py` 會跑自我測試。 |
| `visualize_defects.py` | matplotlib 視覺化，輸出下述兩張圖。 |
| `fig1_interference_physics.png` | 上排：同一顆 defect 只改相對相位 → 亮/雙極/暗；下排：同顆 defect 放到亮/邊/暗 → signature 隨 pattern 變。 |
| `fig2_condition_table.png` | Condition Table (cond2/3/4) + 困難 nuisance 負例，每列顯示 target / reference / difference / label。 |

## Quickstart

```bash
python3 coherent_defect_insertion.py   # 自我測試 (物理 + condition table)
python3 visualize_defects.py           # 產生兩張 PNG
```

```python
import numpy as np, coherent_defect_insertion as cdi
rng = np.random.default_rng(0)
optics = dict(wavelength=266.0, NA=0.90, pixel_pitch=30.0, snr_range=(0.4, 1.2))

# Condition Table 樣本 (condition ∈ {1,2,3,4})
inp, label, meta = cdi.make_sample(target_img, ref_img, condition=2, optics=optics, rng=rng)
# inp: (2,256,256) U-Net 輸入; label: (256,256) probability-map GT

# §4② 困難負例：target/ref 貼不同擾動、label=No
neg_inp, neg_label, _ = cdi.make_nuisance_negative(target_img, ref_img, optics, rng)
```

## 校準（上線前必做）

- `wavelength / NA / pixel_pitch`：從機台規格填。
- `snr_range`：對準實測 4 severe + 66 slight 的訊號分布（目標涵蓋 SNR<1）。
- **`h_ab` 應以「量到的 PSF」當底**（slanted-edge / design-based），Zernike 只作殘餘像差的 augmentation；
  別讓合成 PSF 比真實乾淨太多。
- `sigma_noise`：建議餵入 §3 的 die-to-die per-pixel variance map，讓 SNR 縮放對齊真實局部雜訊。

## 限制與升級路徑

- 這是 **Tier 1（單一 coherent mode）**：已足以修掉「defect 不像 PSF」的 domain gap，投報率最高。
- 唯一物理近似：以 `sqrt(I)` 當背景場模、取樣未知相位。
- **Tier 2（SOCS 部分相干）**：`I = Σ_k α_k |φ_k ⊗ (O_bg + O_defect)|²`，需要物體場 `O_bg`（design/GDS）
  與 coherent-mode 分解 `{φ_k, α_k}`（Cobb 1998）。等 Tier 1 上線、確認 gap 縮小後再投資。
