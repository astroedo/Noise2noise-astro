# Noise2noise-astro

**Learning to denoise astrophotography images without clean ground truth.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astroedo/Noise2noise-astro/blob/main/Noise2Noise_Astro.ipynb)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A self-contained Colab notebook that trains a U-Net to denoise galaxy images using the **Noise2Noise** framework (Lehtinen et al., ICML 2018) — without ever showing the network a clean image.

---

## TL;DR

- Trained a U-Net (~7.7M params) on **Galaxy10 DECaLS** with synthetic Poisson+Gaussian noise pairs.
- Validation PSNR: **18.2 dB → 26.9 dB** (+8.7 dB); SSIM: **0.27 → 0.78**.
- Generalizes to **real amateur astrophotos** (tested on M101 / the Pinwheel Galaxy).
- Single notebook, runs end-to-end on Colab free tier (T4 GPU) in ~15–20 minutes.

---

## Motivation

Supervised denoising assumes (noisy, clean) training pairs. In astrophotography this assumption breaks down: a single exposure is always noisy, and "clean" references only exist as stacked composites of many exposures. **Noise2Noise** shows that if the noise has zero mean and the two observations are independent, training on (noisy_A, noisy_B) pairs converges to the same denoiser as training against the true clean image:

```
E[y₂ | y₁] = x       ⇒     argmin E‖f(y₁) − y₂‖²  =  argmin E‖f(y₁) − x‖²
```

This is exactly the regime that fits real astronomical sensors: Poisson shot noise + Gaussian read noise are both zero-mean, and successive exposures of the same field are independent draws.

---

## Results

### Synthetic validation set (Galaxy10 DECaLS)

| Metric | Noisy input | Denoised | Δ |
|---|---|---|---|
| PSNR (dB) | 18.2 | **26.9** | +8.7 |
| SSIM | 0.27 | **0.78** | +0.51 |

20 epochs, 4000 training images, 128×128 crops, batch size 16, Adam + cosine LR, mixed precision. Trains in ~15 min on a Colab T4.

The model never sees a clean image during training — the loss is computed exclusively between the network output on `noisy_A` and a second independent noisy view `noisy_B`. The clean image is used only as a held-out diagnostic for the metrics above.

### Real-world generalization

Applied to a real amateur astrophoto of **M101 (the Pinwheel Galaxy)** at 4168×3840 px via tiled inference (256-px tiles, 32-px overlap, triangular blend window). The galaxy core, spiral arms, foreground stars, and a faint background galaxy are preserved cleanly; the sky background is smoothed.

The notebook includes an `alpha` blending knob (`alpha=0.6` by default) to tune the aggressiveness, since the model is trained on a higher noise level than typical post-stacking amateur data:

```python
result = alpha * denoised + (1 - alpha) * original
```

---

## What's in the notebook

[`Noise2Noise_Astro.ipynb`](Noise2Noise_Astro.ipynb) — top to bottom, 25 cells:

1. **Setup** — installs `astroNN`, configures device.
2. **Config** — single `CFG` class with every knob.
3. **Dataset** — downloads Galaxy10 DECaLS (~17k galaxy images, 256×256 RGB).
4. **Noise model** — synthetic Poisson + Gaussian, fresh independent draw per `__getitem__` → valid N2N pairs.
5. **Model** — U-Net (encoder-decoder + skip connections), `base=32`, depth 4.
6. **Metrics** — pure-PyTorch PSNR and Gaussian-windowed SSIM (no extra deps).
7. **Training** — MSE loss between `f(noisy_A)` and `noisy_B`, Adam + cosine schedule + AMP.
8. **Curves** — loss / PSNR / SSIM over epochs.
9. **Qualitative results** — side-by-side noisy / denoised / clean grids.
10. **Run on your own astrophoto** — tiled inference with alpha-blended output; saves a PNG.

---

## Running it

**Option A — Google Colab (recommended).** Click the badge at the top, set Runtime → Change runtime type → **GPU (T4)**, then Run all.

**Option B — Local.** Requires PyTorch ≥ 2.1; a CUDA GPU is recommended but not required.

```bash
pip install torch torchvision astroNN h5py matplotlib numpy pillow
jupyter notebook Noise2Noise_Astro.ipynb
```

### Denoising your own image

In section 10 of the notebook:

```python
image_path = 'path/to/your_astrophoto.jpg'
alpha = 0.6        # 1.0 = full denoise, 0.0 = original
```

Runs on arbitrary input size via tiled inference and saves `<name>_denoised_a<alpha>.png`.

---

## Architecture & roadmap

U-Net is the starting point (and the architecture used in the original N2N paper). The training loop is architecture-agnostic — `model(noisy) → denoised` — so the natural progression is:

| Stage | Architecture | Reason |
|---|---|---|
| ✓ v1 | **U-Net** | CNN baseline, paper-faithful, ~7.7M params |
| → next | **NAFNet** (Chen et al., ECCV 2022) | Gated activations + channel attention; SOTA at lower FLOPs than transformers |
| → next | **Restormer** (Zamir et al., CVPR 2022) | Transformer with channel-wise attention (MDTA) — tractable at high resolution |

---

## Limitations

1. **Zero-mean noise assumption.** N2N's guarantee requires noise to be zero-mean and independent across the two observations. Structured noise (JPEG blocking, fixed-pattern sensor noise, hot pixels, amp glow) violates this and will be *learned* by the network rather than removed.
2. **Domain shift on real astrophotos.** The model is trained on Galaxy10 patches (mostly galaxy, dark sky) with synthetic noise. On wide-field amateur photos containing dense star fields, faint nebulosity, or differently-stretched data, it can over-smooth the background. The `alpha` blend partially mitigates this; the proper fix is fine-tuning on real noisy pairs.
3. **Small training subset by default.** `n_images=4000` is set for fast iteration on Colab free tier; PSNR/SSIM improve noticeably with the full 17k images and longer training.
4. **Pixel-wise loss only.** L2 optimizes PSNR directly. A small perceptual term (VGG features) would likely produce sharper textures at the cost of PSNR — useful for visual results, harmful for photometric measurement.

---

## Future work

- **Fine-tune on real noisy pairs.** Two short exposures of the same DSO at the same orientation → genuine N2N training on real sensor noise. This is the regime where the method was designed to shine.
- **Noise2Void** (Krull et al. 2019) for the single-image case (blind-spot masking) — removes the need for paired observations entirely.
- **FITS / 16-bit support** via `astropy` for native astronomical data.
- **Architecture comparison study**: U-Net vs. NAFNet vs. Restormer on the same N2N protocol, with PSNR / SSIM / runtime trade-offs.

---

## References

1. Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., & Aila, T. (2018). **Noise2Noise: Learning Image Restoration without Clean Data.** *ICML 2018.* [arXiv:1803.04189](https://arxiv.org/abs/1803.04189)
2. Ronneberger, O., Fischer, P., & Brox, T. (2015). **U-Net: Convolutional Networks for Biomedical Image Segmentation.** *MICCAI 2015.* [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)
3. Leung, H. W., & Bovy, J. (2019). **Galaxy10 DECaLS dataset.** astroNN documentation: <https://astronn.readthedocs.io/en/latest/galaxy10.html>
4. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). **Image Quality Assessment: From Error Visibility to Structural Similarity.** *IEEE TIP.*
5. Zamir, S. W. et al. (2022). **Restormer: Efficient Transformer for High-Resolution Image Restoration.** *CVPR 2022.* [arXiv:2111.09881](https://arxiv.org/abs/2111.09881)
6. Chen, L., Chu, X., Zhang, X., & Sun, J. (2022). **Simple Baselines for Image Restoration (NAFNet).** *ECCV 2022.* [arXiv:2204.04676](https://arxiv.org/abs/2204.04676)
7. Krull, A., Buchholz, T. O., & Jug, F. (2019). **Noise2Void — Learning Denoising from Single Noisy Images.** *CVPR 2019.* [arXiv:1811.10980](https://arxiv.org/abs/1811.10980)

---

## License

MIT. See [LICENSE](LICENSE).
