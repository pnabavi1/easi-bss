# EASI-BSS

Online blind source separation using EASI with Adam optimizer.

**Pooya Nabavi** (pnabavi@stanford.edu)  
CS229 Machine Learning, Stanford University, Fall 2025

## Introduction

Blind Source Separation (BSS) recovers independent latent sources from observed linear mixtures without prior knowledge of the mixing process. While FastICA provides robust batch separation, online methods such as EASI enable real-time adaptation but suffer from slow convergence with vanilla SGD.

This work introduces **EASI-Adam**, integrating adaptive moment estimation with the EASI natural gradient framework. EASI-Adam achieves **8 dB higher separation SNR** compared to EASI-SGD at 10k iterations on MNIST. Adam's adaptive per-parameter learning rates and momentum enable substantially faster convergence in the online BSS setting, where traditional SGD struggles with poor conditioning and noisy gradients.

## EASI Update Rule

The online stochastic EASI update is:

$$\Delta_k = \mathbf{y}_k\mathbf{y}_k^\top - \mathbf{I} + \mathbf{g}(\mathbf{y}_k)\mathbf{y}_k^\top - \mathbf{y}_k\mathbf{g}(\mathbf{y}_k)^\top$$

$$\mathbf{B}_{k+1} = \mathbf{B}_k - \mu \Delta_k \mathbf{B}_k$$

where $\mathbf{y}_k = \mathbf{B}_k\mathbf{x}_k$ and $\mathbf{g}(\mathbf{y})$ approximates score functions for super-Gaussian and mixed distributions.

EASI-Adam replaces the SGD update with Adam applied to the natural gradient $\mathbf{G}_k = \Delta_k \mathbf{B}_k$.

## Setup

```bash
git clone https://github.com/pnabavi1/easi-bss.git
cd easi-bss
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
from easi_bss import train_easi_adam, prepare_sources, create_mixtures, resolve_permutation_and_scaling

s1, s2, S, H, W = prepare_sources(img1, img2)
X, A = create_mixtures(S)

B, loss_history = train_easi_adam(X, iterations=10000)

Y = B @ X
snr1, snr2, y1, y2 = resolve_permutation_and_scaling(s1, s2, Y[0], Y[1])
```

Or run the experiment:

```bash
python experiments/run_experiment.py --iterations 10000 --save-figures
```

## Results

MNIST digit separation (10k iterations):

| Method | Avg SNR | Memory | Online |
|--------|---------|--------|--------|
| FastICA | 20.60 dB | O(nN + n²) | no |
| EASI-SGD | 3.24 dB | O(n²) | yes |
| EASI-Adam | 11.21 dB | O(n²) | yes |

The 8 dB improvement corresponds to 6× better signal reconstruction in power terms. FastICA achieves higher SNR through batch processing but requires complete dataset access and cannot track time-varying mixtures without full re-computation.

Natural image separation (512×512, 50k iterations):

| Source | SNR |
|--------|-----|
| Cameraman | 28.18 dB |
| Astronaut | 21.21 dB |

## Hyperparameters

| | SGD | Adam |
|---|---|---|
| learning rate | 1e-4 | 1e-3 |
| iterations | 50k | 50k |
| β₁ | - | 0.9 |
| β₂ | - | 0.999 |

## Project layout

```
easi-bss/
├── src/easi_bss/
│   ├── algorithm.py      # train_easi_sgd, train_easi_adam
│   ├── utils.py          # preprocessing, SNR, permutation resolution
│   └── visualization.py
├── experiments/
│   ├── run_experiment.py
│   └── demo.py
└── tests/
```

## References

1. Cardoso & Laheld (1996). Equivariant adaptive source separation. IEEE Trans. Signal Processing, 44(12), 3017-3030.

2. Kingma & Ba (2015). Adam: A method for stochastic optimization. ICLR.

3. Amari (1998). Natural gradient works efficiently in learning. Neural Computation, 10(2), 251-276.

## License

MIT
