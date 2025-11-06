"""
Binaspect - Binaural Audio Spectral Analysis Library

This package provides functions for computing and visualizing various binaural audio features
including Interaural Time Difference (ITD), Interaural Phase Difference (IPD),
Interaural Level Ratio (ILR), and Interaural Level Difference (ILD).
"""

# Import colormap to register custom colormaps with matplotlib
from . import colormap

from .binaspect import (
    ITD_spect,
    IPD_spect,
    ILR_spect,
    ILD_spect,
    ILR_spect_diff,
    ITD_spect_diff,
    ITD_hist,
    ILR_hist,
    ILD_hist,
    ITD_sim,
    ILR_sim,
)

__all__ = [
    "ITD_spect",
    "IPD_spect",
    "ILR_spect",
    "ILD_spect",
    "ILR_spect_diff",
    "ITD_spect_diff",
    "ITD_hist",
    "ILR_hist",
    "ILD_hist",
    "ITD_sim",
    "ILR_sim",
]

__version__ = "1.0.0"
