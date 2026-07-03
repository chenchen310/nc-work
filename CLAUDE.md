# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A **physically-correct virtual-defect generator** for training defect-detection
models on KLA Broadband-Plasma (BBP) optical wafer-inspection images. Input to
the downstream model is a `target`/`reference` image pair (256×256); the model
outputs a probability map. This repo produces the *synthetic training data*.

The whole design turns on one physics insight (read `README.md` for the equations):
in partially-coherent brightfield imaging, a sub-resolution defect's signal in the
difference image is dominated by the **interference** between the defect's scattered
field and the local pattern field. So a real defect looks like the PSF *modulated by
the local pattern* — often asymmetric, bipolar, or dark — **not** a clean symmetric
PSF blob. Pasting an additive intensity PSF (the naive approach) therefore creates a
train/test domain gap. This code instead inserts the defect as a **complex field**,
adds it coherently to `sqrt(I_bg)`, and takes `|·|²`.

## Commands

No build system, test framework, or linter is configured. Deps: `numpy`, `matplotlib`
(Python 3.8+).

```bash
python3 coherent_defect_insertion.py   # self-test: physics + condition-table + broadband asserts
python3 whitened_matched_filter.py     # self-test: whitened MF beats white MF on colored noise
python3 visualize_defects.py           # regenerates fig1 / fig2 / fig3
python3 visualize_matched_filter.py    # regenerates fig4 (whitened matched filter)
```

Each module's `__main__` block **is** its test suite (`coherent_defect_insertion.py`:
interference sign-flip, condition-table labels, broadband ring damping;
`whitened_matched_filter.py`: detection-SNR / whitened-beats-white). Extend those
asserts rather than adding a framework. There is no single-test runner; run the whole file.

## Architecture

Two modules; `visualize_defects.py` imports `coherent_defect_insertion` as `cdi`.

**`coherent_defect_insertion.py`** — the generator library. Pipeline:
1. `aberrated_pupil(...)` builds a complex pupil `P(f)=A(ρ)·exp(i2π·ΣcⱼZⱼ)` on a
   centered FFT grid, with Zernike (Noll-indexed) defocus/aberration in **waves** and
   an optional sub-pixel shift baked in as a linear pupil phase.
2. `amplitude_psf(P)` → complex amplitude PSF (ASF) via centered inverse FFT.
3. `insert_defect(...)` is the core: `E_total = sqrt(I_bg)·exp(iφ_bg) + a_d·h`, then
   `I_new = |E_total|²`. This is what reproduces the interference term.
4. `make_sample(...)` implements the **Condition Table** that maps (defect on target?,
   defect on ref?) → label: cond1 none→No, cond2 target-only→Defect(+Gaussian label),
   cond3 ref-only→No, cond4 both-identical→No (common-mode). Returns a 2-channel
   `(2,256,256)` input stack + a `(256,256)` probability-map ground truth.
5. `make_nuisance_negative(...)` is the deliberate complement to the condition table:
   it pastes **different** perturbations on target vs. reference (label=No) to mimic
   process-variation nuisance. The condition-table negatives are all "easy"
   (common-mode); this supplies the hard negatives the real tool actually sees.

**SNR scaling** (`_scale_to_snr`) sets defect amplitude by bisection on the *local*
RMS of the true (nonlinear) ΔI vs. a local noise σ — not a closed-form scale. This
avoids over-amplifying when the relative phase ≈ π/2 makes the interference term
vanish and the weak self-term dominates (a real, not artificial, effect).

## Non-obvious constraints & gotchas

- **`sqrt(I_bg)` + sampled `φ_bg` is the one physical approximation.** We only measure
  intensity, so the background field phase is unknown; it is *sampled* rather than
  reconstructed. For a data generator this is a feature (spans bright↔dark), not a bug.
- **`pixel_pitch` ≠ optical resolution.** The tool's "30 nm" is sampling; optical
  resolution is `λ/(2·NA)` (~148 nm at the demo's λ=266 nm, NA=0.90). Keep this
  distinction; the sampling assert is `oversample = (NA/λ)/(1/(2·Δx)) < 1`.
- **`h` should ideally be a *measured* PSF**, with Zernike used only as residual-aberration
  augmentation — don't ship synthetic PSFs that are cleaner than the real tool.
- **`snr_range` / `sigma_noise` must be calibrated** to real signal/noise; feed a real
  die-to-die per-pixel variance map into `sigma_noise` when available.
- **matplotlib has no CJK font here.** All in-figure text must be ASCII/Greek
  (Δ, π render; CJK does not). Chinese belongs in code comments / `README.md`, never
  in figure titles/labels. Prose/README are in Traditional Chinese by convention.

## Scope / roadmap

This is **Tier 1** (single coherent mode) — enough to close the "defect ≠ PSF" domain
gap. **Tier 2** (SOCS partial coherence, `I=Σₖαₖ|φₖ⊗(O_bg+O_defect)|²`) needs the
object field `O_bg` from design/GDS and a coherent-mode decomposition; only pursue it
after Tier 1 is validated. The whitened matched-filter detection front end now lives in
`whitened_matched_filter.py` (spectral whitening by the estimated die-to-die noise PSD →
matched-subspace `{Re(h), Im(h)}` statistic `T = z_re² + z_im²`); its `score` map is
meant to be fed as an extra U-Net input channel. A remaining extension is full spatial
whitening by a non-stationary per-pixel covariance (current whitening is spectral/stationary).
