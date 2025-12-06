#!/usr/bin/env python3
"""
BSS experiment on MNIST: compares EASI-SGD, EASI-Adam, FastICA.

Usage: python run_experiment.py [--iterations N] [--save-figures]
"""

import argparse
import time
import numpy as np
from pathlib import Path

# Import EASI modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from easi_bss import (
    train_easi_sgd,
    train_easi_adam,
    prepare_sources,
    create_mixtures,
    resolve_permutation_and_scaling,
    display_images,
    plot_histogram_with_pdf,
    plot_scatter,
    plot_scatter_comparison,
    plot_loss_curves,
    plot_method_comparison,
    plot_convergence_comparison,
)

# External imports
from sklearn.decomposition import FastICA
from keras import datasets as dt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run EASI blind source separation experiment on MNIST"
    )
    parser.add_argument(
        "--iterations", type=int, default=10000,
        help="Number of training iterations (default: 10000)"
    )
    parser.add_argument(
        "--sgd-lr", type=float, default=1e-4,
        help="Learning rate for SGD (default: 1e-4)"
    )
    parser.add_argument(
        "--adam-lr", type=float, default=1e-3,
        help="Learning rate for Adam (default: 1e-3)"
    )
    parser.add_argument(
        "--save-figures", action="store_true",
        help="Save figures to figures/ directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    fig_dir = Path(__file__).parent.parent / "figures"
    if args.save_figures:
        fig_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("EASI BSS Experiment")
    print(f"iters={args.iterations}, sgd_lr={args.sgd_lr}, adam_lr={args.adam_lr}")
    print("=" * 60)
    
    # load data
    print("\n[1] Loading MNIST...")
    (train_data, _), (test_data, _) = dt.mnist.load_data()
    
    img1 = test_data[0]  # 7
    img2 = test_data[4]  # 4
    s1, s2, S, H, W = prepare_sources(img1, img2)
    N = S.shape[1]
    print(f"    {H}x{W} = {N} pixels")
    
    s1_img = s1.reshape(H, W)
    s2_img = s2.reshape(H, W)
    save_path = str(fig_dir / "sources.png") if args.save_figures else None
    display_images([s1_img, s2_img], ['$s_1$', '$s_2$'], save_path=save_path)
    
    # mix
    print("\n[2] Mixing...")
    X, A = create_mixtures(S)
    x1, x2 = X[0], X[1]
    print(f"    A = {A.tolist()}, cond = {np.linalg.cond(A):.2f}")
    
    save_path = str(fig_dir / "mixtures.png") if args.save_figures else None
    display_images([x1.reshape(H,W), x2.reshape(H,W)], ['$x_1$', '$x_2$'], save_path=save_path)
    
    # EASI-SGD
    print(f"\n[3] EASI-SGD ({args.iterations} iters)...")
    t0 = time.time()
    B_sgd, loss_sgd = train_easi_sgd(X, learning_rate=args.sgd_lr, iterations=args.iterations)
    sgd_time = time.time() - t0
    
    Y_sgd = B_sgd @ X
    SNR1_sgd, SNR2_sgd, y1_sgd, y2_sgd = resolve_permutation_and_scaling(s1, s2, Y_sgd[0], Y_sgd[1])
    avg_sgd = (SNR1_sgd + SNR2_sgd) / 2
    print(f"    {sgd_time:.1f}s, SNR = {avg_sgd:.2f} dB")
    
    save_path = str(fig_dir / "loss_sgd.png") if args.save_figures else None
    plot_loss_curves(loss_sgd, title="EASI-SGD", save_path=save_path)
    
    # EASI-Adam
    print(f"\n[4] EASI-Adam ({args.iterations} iters)...")
    t0 = time.time()
    B_adam, loss_adam = train_easi_adam(X, learning_rate=args.adam_lr, iterations=args.iterations)
    adam_time = time.time() - t0
    
    Y_adam = B_adam @ X
    SNR1_adam, SNR2_adam, y1_adam, y2_adam = resolve_permutation_and_scaling(s1, s2, Y_adam[0], Y_adam[1])
    avg_adam = (SNR1_adam + SNR2_adam) / 2
    print(f"    {adam_time:.1f}s, SNR = {avg_adam:.2f} dB")
    
    save_path = str(fig_dir / "loss_adam.png") if args.save_figures else None
    plot_loss_curves(loss_adam, title="EASI-Adam", save_path=save_path)
    
    # FastICA
    print("\n[5] FastICA...")
    t0 = time.time()
    ica = FastICA(n_components=2, max_iter=200, random_state=args.seed)
    Y_ica = ica.fit_transform(X.T).T
    ica_time = time.time() - t0
    
    SNR1_ica, SNR2_ica, y1_ica, y2_ica = resolve_permutation_and_scaling(s1, s2, Y_ica[0], Y_ica[1])
    avg_ica = (SNR1_ica + SNR2_ica) / 2
    print(f"    {ica_time:.1f}s, SNR = {avg_ica:.2f} dB")
    
    # summary
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'SNR (dB)':<12} {'Time (s)':<10}")
    print("-" * 40)
    print(f"{'FastICA':<15} {avg_ica:<12.2f} {ica_time:<10.1f}")
    print(f"{'EASI-SGD':<15} {avg_sgd:<12.2f} {sgd_time:<10.1f}")
    print(f"{'EASI-Adam':<15} {avg_adam:<12.2f} {adam_time:<10.1f}")
    print("=" * 60)
    
    # plots
    print("\n[6] Plots...")
    results = {
        'original': {'s1': s1, 's2': s2},
        'mixed': {'x1': x1, 'x2': x2},
        'fastica': {'y1': y1_ica, 'y2': y2_ica, 'snr1': SNR1_ica, 'snr2': SNR2_ica},
        'easi_sgd': {'y1': y1_sgd, 'y2': y2_sgd, 'snr1': SNR1_sgd, 'snr2': SNR2_sgd},
        'easi_adam': {'y1': y1_adam, 'y2': y2_adam, 'snr1': SNR1_adam, 'snr2': SNR2_adam}
    }
    
    save_path = str(fig_dir / "comparison.png") if args.save_figures else None
    plot_method_comparison(results, H, W, save_path=save_path)
    
    snr_results = {
        'FastICA': {'avg': avg_ica},
        'EASI-SGD': {'avg': avg_sgd},
        'EASI-Adam': {'avg': avg_adam}
    }
    save_path = str(fig_dir / "convergence.png") if args.save_figures else None
    plot_convergence_comparison(loss_sgd, loss_adam, snr_results, save_path=save_path)
    
    save_path = str(fig_dir / "independence.png") if args.save_figures else None
    plot_scatter_comparison(s1, s2, x1, x2, y1_adam, y2_adam, save_path=save_path)
    
    print("\nDone.")
    if args.save_figures:
        print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
