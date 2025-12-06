"""Plotting functions for BSS experiments."""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

from .utils import kl_divergence


def configure_plots():
    """Set some reasonable defaults."""
    plt.rcParams["figure.figsize"] = (8, 5)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.dpi"] = 100


def display_images(images, titles, figsize=(12, 5), cmap='gray', save_path=None, show=True):
    """Show images side by side."""
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(f'{title}\n{img.shape}, [{img.min():.3f}, {img.max():.3f}]')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_histogram_with_pdf(x, num_bins=100, title_prefix="s", save_path=None, show=True):
    """Histogram + estimated PDF."""
    x = x.flatten()
    N = len(x)
    hist, edges = np.histogram(x, bins=num_bins)
    delta = (edges[-1] - edges[0]) / num_bins
    pdf = hist / (N * delta)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    ax1.bar(centers, hist, width=delta, color='steelblue', edgecolor='none')
    ax1.set_xlabel(title_prefix)
    ax1.set_ylabel("Count")
    ax1.set_title(f"Histogram of {title_prefix}")
    ax1.grid(alpha=0.3)

    ax2.plot(centers, pdf, lw=2, color='steelblue')
    ax2.fill_between(centers, pdf, alpha=0.3)
    ax2.set_xlabel(title_prefix)
    ax2.set_ylabel("Density")
    ax2.set_title(f"PDF of {title_prefix}")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_pdf_comparison(original, recovered, num_bins=100, title_prefix="s", save_path=None, show=True):
    """Compare original vs recovered PDFs. Returns (fig, kl_div)."""
    orig = original.flatten()
    rec = recovered.flatten()
    N = len(orig)

    edges = np.linspace(min(orig.min(), rec.min()), max(orig.max(), rec.max()), num_bins + 1)
    delta = edges[1] - edges[0]
    centers = (edges[:-1] + edges[1:]) / 2

    h_orig, _ = np.histogram(orig, bins=edges)
    h_rec, _ = np.histogram(rec, bins=edges)
    kl = kl_divergence(h_orig, h_rec)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    w = delta * 0.4
    ax1.bar(centers - w/2, h_orig, width=w, alpha=0.7, label='Original', color='blue')
    ax1.bar(centers + w/2, h_rec, width=w, alpha=0.7, label='Recovered', color='red')
    ax1.set_xlabel(title_prefix)
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram Comparison")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(centers, h_orig/(N*delta), lw=2, label='Original', color='blue')
    ax2.plot(centers, h_rec/(N*delta), lw=2, label='Recovered', color='red', ls='--')
    ax2.set_xlabel(title_prefix)
    ax2.set_ylabel("Density")
    ax2.set_title(f"PDF Comparison (KL = {kl:.4f})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, kl


def plot_scatter(x1, x2, title, xlabel, ylabel, save_path=None, show=True):
    """Simple scatter plot."""
    plt.figure(figsize=(6, 6))
    plt.scatter(x1.flatten(), x2.flatten(), s=2, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return plt.gcf()


def plot_scatter_comparison(s1, s2, x1, x2, y1, y2, save_path=None, show=True):
    """Original -> mixed -> recovered scatter plots."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(s1.flatten(), s2.flatten(), s=2, alpha=0.5, color='blue')
    axes[0].set_xlabel('$s_1$')
    axes[0].set_ylabel('$s_2$')
    axes[0].set_title('Original (rectangular = independent)')
    axes[0].axis('equal')
    axes[0].grid(alpha=0.3)

    axes[1].scatter(x1.flatten(), x2.flatten(), s=2, alpha=0.5, color='red')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    axes[1].set_title('Mixed (elliptical = correlated)')
    axes[1].axis('equal')
    axes[1].grid(alpha=0.3)

    axes[2].scatter(y1.flatten(), y2.flatten(), s=2, alpha=0.5, color='green')
    axes[2].set_xlabel('$\\hat{s}_1$')
    axes[2].set_ylabel('$\\hat{s}_2$')
    axes[2].set_title('Recovered')
    axes[2].axis('equal')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_loss_curves(loss_history, title="EASI Training", save_path=None, show=True):
    """Plot loss curves from training."""
    iters = [h['iteration'] for h in loss_history]
    total = [h['total_loss'] for h in loss_history]
    decorr = [h['decorrelation_loss'] for h in loss_history]
    det = [h['log_det_loss'] for h in loss_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(iters, total, 'b-', lw=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(iters, decorr, 'r-', lw=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Decorrelation Loss')
    axes[0, 1].set_title('$||E[yy^T] - I||^2$')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(iters, det, 'g-', lw=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Log Det Loss')
    axes[1, 0].set_title('$-\\log|\\det(B)|$')
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(iters, np.array(total)/total[0], 'b-', lw=2, label='Total', alpha=0.7)
    axes[1, 1].plot(iters, np.array(decorr)/decorr[0], 'r-', lw=2, label='Decorr', alpha=0.7)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Normalized')
    axes[1, 1].set_title('Convergence')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


COLORS = {'fastica': '#3498DB', 'easi_sgd': '#E74C3C', 'easi_adam': '#27AE60'}


def plot_method_comparison(results, H, W, save_path=None, show=True):
    """Visual comparison grid: original, mixed, and all methods."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    # row 1: s1
    axes[0, 0].imshow(results['original']['s1'].reshape(H, W), cmap='gray')
    axes[0, 0].set_title('Original $s_1$', fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(results['mixed']['x1'].reshape(H, W), cmap='gray')
    axes[0, 1].set_title('Mixed $x_1$', fontweight='bold')
    axes[0, 1].axis('off')

    col = 2
    for method in ['fastica', 'easi_sgd', 'easi_adam']:
        if method in results:
            y1 = results[method]['y1'].reshape(H, W)
            snr = results[method]['snr1']
            label = {'fastica': 'FastICA', 'easi_sgd': 'EASI-SGD', 'easi_adam': 'EASI-Adam'}[method]
            axes[0, col].imshow(y1, cmap='gray')
            axes[0, col].set_title(f'{label}\nSNR={snr:.2f} dB', fontweight='bold', color=COLORS[method])
            axes[0, col].axis('off')
            col += 1

    # row 2: s2
    axes[1, 0].imshow(results['original']['s2'].reshape(H, W), cmap='gray')
    axes[1, 0].set_title('Original $s_2$', fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(results['mixed']['x2'].reshape(H, W), cmap='gray')
    axes[1, 1].set_title('Mixed $x_2$', fontweight='bold')
    axes[1, 1].axis('off')

    col = 2
    for method in ['fastica', 'easi_sgd', 'easi_adam']:
        if method in results:
            y2 = results[method]['y2'].reshape(H, W)
            snr = results[method]['snr2']
            label = {'fastica': 'FastICA', 'easi_sgd': 'EASI-SGD', 'easi_adam': 'EASI-Adam'}[method]
            axes[1, col].imshow(y2, cmap='gray')
            axes[1, col].set_title(f'{label}\nSNR={snr:.2f} dB', fontweight='bold', color=COLORS[method])
            axes[1, col].axis('off')
            col += 1

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig


def plot_convergence_comparison(loss_sgd, loss_adam, snr_results, save_path=None, show=True):
    """SGD vs Adam: loss curves + SNR bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # loss curves
    ax1.plot([h['iteration'] for h in loss_sgd], [h['total_loss'] for h in loss_sgd],
             'o-', color=COLORS['easi_sgd'], lw=2, ms=4, markevery=10, label='SGD', alpha=0.8)
    ax1.plot([h['iteration'] for h in loss_adam], [h['total_loss'] for h in loss_adam],
             's-', color=COLORS['easi_adam'], lw=2, ms=4, markevery=10, label='Adam', alpha=0.8)
    ax1.set_xlabel('Iteration', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Convergence', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # SNR bars
    methods = list(snr_results.keys())
    vals = [snr_results[m]['avg'] for m in methods]
    colors = [COLORS.get(m.lower().replace('-', '_').replace(' ', '_'), '#666') for m in methods]
    bars = ax2.bar(methods, vals, color=colors, edgecolor='black', lw=1.5, alpha=0.8)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.2f}', ha='center', fontweight='bold')
    ax2.set_ylabel('Avg SNR (dB)', fontweight='bold')
    ax2.set_title('Performance', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig
