import os

import argparse

import random
import numpy as np

import librosa
import librosa.display

import tensorflow as tf

from utils.dataset import Dataset, N_FFT, FRAMES_PER_SECOND, STFT_SCALE, HOP_LENGTH
from utils.recordings import write_wave, TRAINING_LENGTH, SAMPLE_RATE
from utils.cirm import generate_mask, apply_mask

from models.unet import get_model

import matplotlib.pyplot as plt

RANDOM_SEED = 119

DATASET_ROOT = "#TODO"
SNAPSHOT_FILE_NAME = "best_snapshot.ckpt"

MAX_OUTS = 10000


def reconstruct_stft(real, imag):
    re = real * STFT_SCALE
    im = imag * STFT_SCALE

    extended_re = np.zeros((re.shape[0] + 1, re.shape[1]), dtype=re.dtype)
    extended_re[: re.shape[0]] = re

    extended_im = np.zeros((im.shape[0] + 1, im.shape[1]), dtype=im.dtype)
    extended_im[: im.shape[0]] = im

    stft = extended_re + 1j * extended_im
    return stft


def add_spectrogram(spectrogram, axis, title):
    S, phase = librosa.magphase(spectrogram)
    colorbar = librosa.display.specshow(
        librosa.amplitude_to_db(S, amin=1e-10, top_db=150.0),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        y_axis="linear",
        x_axis="time",
        ax=axis,
        vmin=-80,
        vmax=20,
        cmap="magma",
    )
    axis.set(title=title)
    return colorbar


def reconstruct_and_save_wave(stft, idx, name, out_dir):
    wave = librosa.istft(stft, hop_length=HOP_LENGTH)
    filename = os.path.join(out_dir, f"{str(idx).zfill(4)}_{name}.wav")
    write_wave(filename, wave, SAMPLE_RATE)


def reconstruct_and_visualize(
    orig_mask, mixed_stft, reconst_mask, idx, out_dir, visualize
):
    y_true_mask_real = orig_mask[:, :, 2]
    y_true_mask_imag = orig_mask[:, :, 3]

    stft_orig = reconstruct_stft(
        *apply_mask(
            y_true_mask_real, y_true_mask_imag, mixed_stft[:, :, 0], mixed_stft[:, :, 1]
        )
    )

    mix_mask_r, mix_mask_i = generate_mask(
        mixed_stft[:, :, 0],
        mixed_stft[:, :, 1],
        mixed_stft[:, :, 0],
        mixed_stft[:, :, 1],
    )
    stft_mixed = reconstruct_stft(
        *apply_mask(mix_mask_r, mix_mask_i, mixed_stft[:, :, 0], mixed_stft[:, :, 1])
    )

    reconst_mask_bin = reconst_mask[:, :, 0]
    # Perfect mask
    #    reconst_mask_bin = orig_mask[:, :, 0]
    reconst_mask_real = reconst_mask[:, :, 1]
    reconst_mask_imag = reconst_mask[:, :, 2]

    reconst_mask_bin = (reconst_mask_bin >= 0.5) * 1.0
    reconst_mask_real *= reconst_mask_bin
    reconst_mask_imag *= reconst_mask_bin
    del reconst_mask

    stft_reconst = reconstruct_stft(
        *apply_mask(
            reconst_mask_real,
            reconst_mask_imag,
            mixed_stft[:, :, 0],
            mixed_stft[:, :, 1],
        )
    )

    reconstruct_and_save_wave(stft_orig, idx, "orig", out_dir)
    reconstruct_and_save_wave(stft_mixed, idx, "mixed", out_dir)
    reconstruct_and_save_wave(stft_reconst, idx, "reconst", out_dir)

    if not visualize:
        return

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(20, 10))

    colorbar = add_spectrogram(stft_orig, ax[0], "Original salient sample")
    colorbar = add_spectrogram(stft_mixed, ax[1], "Salient sample in background")
    colorbar = add_spectrogram(stft_reconst, ax[2], "Reconstructed salient sample")
    fig.colorbar(colorbar, ax=ax, fraction=0.05)

    for ax_i in ax:
        ax_i.label_outer()

    plt.savefig(os.path.join(out_dir, f"{str(idx).zfill(4)}.png"))

    fig.clear()
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test of anomalous sounds detection demo"
    )
    parser.add_argument(
        "--snr",
        type=int,
        required=True,
        help="Set to velue GTE 100 to test with pure salient sound",
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--visualize", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = Dataset(DATASET_ROOT, forced_snr=args.snr)

    model = get_model(N_FFT // 2, TRAINING_LENGTH * FRAMES_PER_SECOND, 2)
    model.load_weights(os.path.join(args.model_dir, SNAPSHOT_FILE_NAME))
    model.summary()

    x_test, y_test = dataset.get_test_data()

    for test_sample_idx in range(min(MAX_OUTS, x_test.shape[0])):
        y_pred = model(x_test[test_sample_idx, None]).numpy()[0]
        reconstruct_and_visualize(
            y_test[test_sample_idx],
            x_test[test_sample_idx],
            y_pred,
            test_sample_idx,
            args.out_dir,
            args.visualize,
        )


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    main()
