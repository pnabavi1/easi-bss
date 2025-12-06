"""EASI-BSS: online blind source separation with Adam."""

from .algorithm import (
    train_easi_sgd,
    train_easi_adam,
    separate_sources,
    easi_score_function,
    tanh_score_function,
    compute_easi_loss,
)

from .utils import (
    prepare_sources,
    create_mixtures,
    resolve_permutation_and_scaling,
    snr_db,
)

from .visualization import (
    plot_loss_curves,
    plot_method_comparison,
    plot_convergence_comparison,
)

__version__ = "1.0.0"
__author__ = "Pooya Nabavi"
