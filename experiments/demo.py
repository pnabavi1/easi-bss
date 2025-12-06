#!/usr/bin/env python3
"""Quick demo of EASI blind source separation on MNIST."""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easi_bss import (
    train_easi_adam,
    prepare_sources,
    create_mixtures,
    resolve_permutation_and_scaling,
    display_images,
)

from sklearn.decomposition import FastICA
from keras import datasets as dt


def main():
    print("EASI Blind Source Separation - Quick Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Load MNIST
    print("\n1. Loading data...")
    (_, _), (test_data, _) = dt.mnist.load_data()
    
    # Prepare sources from two different digits
    img1, img2 = test_data[0], test_data[4]
    s1, s2, S, H, W = prepare_sources(img1, img2)
    
    # Create mixtures
    print("2. Creating mixtures...")
    X, A = create_mixtures(S)
    print(f"   Mixing matrix:\n   {A}")
    
    # Run EASI-Adam
    print("\n3. Running EASI-Adam (5000 iterations)...")
    B_adam, _ = train_easi_adam(X, iterations=5000, verbose=False)
    Y_adam = B_adam @ X
    snr1, snr2, y1, y2 = resolve_permutation_and_scaling(s1, s2, Y_adam[0], Y_adam[1])
    print(f"   SNR: {(snr1+snr2)/2:.2f} dB")
    
    # Run FastICA for comparison
    print("\n4. Running FastICA baseline...")
    ica = FastICA(n_components=2, random_state=42)
    Y_ica = ica.fit_transform(X.T).T
    snr1_ica, snr2_ica, y1_ica, y2_ica = resolve_permutation_and_scaling(s1, s2, Y_ica[0], Y_ica[1])
    print(f"   SNR: {(snr1_ica+snr2_ica)/2:.2f} dB")
    
    # Display results
    print("\n5. Displaying results...")
    images = [
        s1.reshape(H, W),
        s2.reshape(H, W),
        X[0].reshape(H, W),
        X[1].reshape(H, W),
        y1.reshape(H, W),
        y2.reshape(H, W),
    ]
    titles = [
        'Source $s_1$', 'Source $s_2$',
        'Mixed $x_1$', 'Mixed $x_2$',
        f'EASI $\\hat{{s}}_1$ ({snr1:.1f}dB)', f'EASI $\\hat{{s}}_2$ ({snr2:.1f}dB)'
    ]
    display_images(images[:2], titles[:2], figsize=(8, 4))
    display_images(images[2:4], titles[2:4], figsize=(8, 4))
    display_images(images[4:], titles[4:], figsize=(8, 4))
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
