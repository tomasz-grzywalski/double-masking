import os

import random
import string

import numpy as np
from scipy import ndimage

import librosa
import librosa.display

import matplotlib.pyplot as plt

DUMP_DIR = "./DUMPS/"


def add_db(db, axis, title, hop_length, sample_rate):
    colorbar = librosa.display.specshow(
        db,
        sr=sample_rate,
        hop_length=hop_length,
        y_axis="linear",
        x_axis="time",
        ax=axis,
        vmax=20,
        vmin=-80,
        cmap="magma",
    )
    axis.set(title=title)
    return colorbar


def add_spectrogram(spectrogram, axis, title, hop_length, sample_rate):
    S, phase = librosa.magphase(spectrogram)
    db = librosa.amplitude_to_db(S, amin=1e-10, top_db=150.0)
    return add_db(db, axis, title, hop_length, sample_rate)


def visualize_mix(mixture, fgnd_after_mask, mask, margin, hop_length, sample_rate):
    fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(20, 13))

    colorbar = add_spectrogram(
        mixture,
        ax[0],
        "Mixture",
        hop_length,
        sample_rate,
    )
    fig.colorbar(colorbar, ax=ax[0], fraction=0.05)

    colorbar = add_spectrogram(
        fgnd_after_mask,
        ax[1],
        "Foreground sound extracted using perfect mask",
        hop_length,
        sample_rate,
    )
    fig.colorbar(colorbar, ax=ax[1], fraction=0.05)

    mask = mask * 100 - 80
    _ = add_db(
        mask,
        ax[2],
        "Ground truth Region of Interest binary mask",
        hop_length,
        sample_rate,
    )
    fig.colorbar(colorbar, ax=ax[2], fraction=0.05)

    margin = margin * 100 - 80
    _ = add_db(
        margin,
        ax[3],
        "Region of Interest binary mask margin",
        hop_length,
        sample_rate,
    )
    fig.colorbar(colorbar, ax=ax[3], fraction=0.05)

    for ax_i in ax:
        ax_i.label_outer()

    random_filename = "".join(random.choice(string.ascii_letters) for i in range(10))
    plt.savefig(os.path.join(DUMP_DIR, f"{random_filename}.png"))

    fig.clear()
    plt.close(fig)
