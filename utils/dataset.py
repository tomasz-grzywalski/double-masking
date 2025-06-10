import os

import random

import numpy as np
import pandas as pd
from scipy import ndimage
import librosa

from utils.recordings import TRAINING_LENGTH, SAMPLE_RATE, get_background, mixin
from utils.cirm import generate_mask, apply_mask
from utils.visualizations import visualize_mix

BACKGROUND_DIR, FOREGROUND_DIR = "Background", "Foreground"
MAX_FFBF_ID = 51598
MAX_FFBF_N = 5
FGND_MAX_SOUNDS = 5

TEST_VAL_RECORDINGS = 1500

N_FFT = 512
FRAMES_PER_SECOND = 100
HOP_LENGTH = SAMPLE_RATE // FRAMES_PER_SECOND
STFT_SCALE = 40.0

VISUALIZE_DATASET = False


class Dataset:
    def __init__(self, data_root, forced_snr=None):
        self._data_root = data_root
        self._forced_snr = forced_snr

        self._train_bgnd = self.get_bgnd()
        self._train_fgnd = self.get_fgnd()

        (
            self.x_test,
            self.y_test,
            num_bgnd,
            num_fgnd,
            num_pos,
        ) = self.get_test_val_samples()
        print(
            f"Test BGND parts: {num_bgnd} FGND sounds: {num_fgnd} Positive samples: {num_pos}"
        )

        (
            self.x_val,
            self.y_val,
            num_bgnd,
            num_fgnd,
            num_pos,
        ) = self.get_test_val_samples()
        print(
            f"Val BGND parts: {num_bgnd} FGND sounds: {num_fgnd} Positive samples: {num_pos}"
        )

        print(
            f"Train BGND parts: {len(self._train_bgnd)} FGND sounds: {len(self._train_fgnd)}"
        )

    def get_dir_name(self, original_rec_id):
        return f"{(original_rec_id // 100):04d}"

    def get_bgnd_file_name(self, original_rec_id, original_part_id):
        dir_name = self.get_dir_name(original_rec_id)
        file_name = f"{original_rec_id:05d}_{original_part_id}.wav"
        return os.path.join(self._data_root, BACKGROUND_DIR, dir_name, file_name)

    def get_fgnd_file_name(self, original_rec_id, original_part_id, original_sound_id):
        dir_name = self.get_dir_name(original_rec_id)
        file_name = (
            f"{original_rec_id:05d}_{original_part_id}_{original_sound_id:02d}.wav"
        )
        return os.path.join(self._data_root, FOREGROUND_DIR, dir_name, file_name)

    def get_bgnd(self):
        root = os.path.join(self._data_root, BACKGROUND_DIR)

        bgnd = list()
        for original_rec_id in range(MAX_FFBF_ID + 1):
            for original_part_id in range(1, MAX_FFBF_N + 1):
                full_file_name = self.get_bgnd_file_name(
                    original_rec_id, original_part_id
                )
                if os.path.exists(full_file_name):
                    bgnd.append([original_rec_id, original_part_id])

        bgnd = pd.DataFrame(bgnd, columns=["rec_id", "part_id"])

        original_recs = bgnd["rec_id"].unique()
        print(
            f"There are {len(bgnd)} BGND parts from {len(original_recs)} original recordings"
        )
        return bgnd

    def get_fgnd(self):
        root = os.path.join(self._data_root, FOREGROUND_DIR)

        fgnd = list()
        for original_rec_id in range(MAX_FFBF_ID + 1):
            for original_part_id in range(1, MAX_FFBF_N + 1):
                for original_sound_id in range(1, FGND_MAX_SOUNDS + 1):
                    full_file_name = self.get_fgnd_file_name(
                        original_rec_id, original_part_id, original_sound_id
                    )
                    if os.path.exists(full_file_name):
                        fgnd.append(
                            [original_rec_id, original_part_id, original_sound_id]
                        )
                    else:
                        break

        fgnd = pd.DataFrame(fgnd, columns=["rec_id", "part_id", "sound_id"])

        original_recs = fgnd["rec_id"].unique()
        print(
            f"There are {len(fgnd)} FGND sounds from {len(original_recs)} original recordings"
        )
        return fgnd

    def get_test_val_samples(self):
        bgnd, fgnd = self.get_some_from_train(TEST_VAL_RECORDINGS)
        x, y, pos = self.get_samples(bgnd, fgnd)
        return x, y, len(bgnd), len(fgnd), pos

    def get_some_from_train(self, num_recordings):
        original_bgnd_rec_ids = self._train_bgnd["rec_id"].unique()
        original_fgnd_rec_ids = self._train_fgnd["rec_id"].unique()

        rec_ids = list(set(list(original_bgnd_rec_ids) + list(original_fgnd_rec_ids)))
        random.shuffle(rec_ids)
        rec_ids = rec_ids[:num_recordings]

        bgnd = self._train_bgnd[self._train_bgnd["rec_id"].isin(rec_ids)]
        self._train_bgnd = self._train_bgnd.drop(
            self._train_bgnd.index[self._train_bgnd["rec_id"].isin(rec_ids)]
        )

        fgnd = self._train_fgnd[self._train_fgnd["rec_id"].isin(rec_ids)]
        self._train_fgnd = self._train_fgnd.drop(
            self._train_fgnd.index[self._train_fgnd["rec_id"].isin(rec_ids)]
        )

        return bgnd, fgnd

    def get_samples(self, bgnd, fgnd):
        bgnd = bgnd.sample(frac=1).reset_index(drop=True)
        fgnd = fgnd.sample(frac=1).reset_index(drop=True)
        fgnd_pointer = 0

        inputs = np.empty(
            (len(bgnd), N_FFT // 2, TRAINING_LENGTH * FRAMES_PER_SECOND, 2),
            dtype=np.float32,
        )
        outputs = np.empty(
            (len(bgnd), N_FFT // 2, TRAINING_LENGTH * FRAMES_PER_SECOND, 4),
            dtype=np.float32,
        )

        pos_samples = 0
        for idx, bgnd_row in bgnd.iterrows():
            (
                inputs[idx],
                outputs[idx],
                fgnd_pointer,
                pos_sample,
            ) = self.prepare_recording(bgnd_row, fgnd, fgnd_pointer)
            pos_samples += 1 if pos_sample else 0

        return inputs, outputs, pos_samples

    def iterate_train_samples(self, batch_size):
        bgnd = self._train_bgnd.sample(frac=1).reset_index(drop=True)
        fgnd = self._train_fgnd.sample(frac=1).reset_index(drop=True)
        fgnd_pointer = 0

        inputs = np.empty(
            (batch_size, N_FFT // 2, TRAINING_LENGTH * FRAMES_PER_SECOND, 2),
            dtype=np.float32,
        )
        outputs = np.empty(
            (batch_size, N_FFT // 2, TRAINING_LENGTH * FRAMES_PER_SECOND, 4),
            dtype=np.float32,
        )

        num_batches = len(bgnd) // batch_size
        for batch_idx in range(num_batches):
            for sample_id in range(batch_size):
                bgnd_row = bgnd.iloc[batch_idx * batch_size + sample_id]
                (
                    inputs[sample_id],
                    outputs[sample_id],
                    fgnd_pointer,
                    _,
                ) = self.prepare_recording(bgnd_row, fgnd, fgnd_pointer)

            yield inputs, outputs

    def prepare_recording(self, bgnd_row, fgnd, fgnd_ptr):
        bgnd_file_name = self.get_bgnd_file_name(
            bgnd_row["rec_id"], bgnd_row["part_id"]
        )
        mixture = get_background(bgnd_file_name)
        pure_fgnd = np.zeros_like(mixture)

        # If SNR is GTE 100 we enter a special mode with no background, but we need to get realistic silent sound from mixin
        # So for a moment we pretend that SNR is 0
        snr = (
            self._forced_snr
            if (self._forced_snr is None or self._forced_snr < 100)
            else 0
        )

        # 50% chance we will return pure background
        if random.choice([0, 1]):
            fgnr_row = fgnd.loc[fgnd_ptr]
            fgnd_ptr = 0 if fgnd_ptr + 1 == len(fgnd) else fgnd_ptr + 1
            fgnd_file_name = self.get_fgnd_file_name(
                fgnr_row["rec_id"], fgnr_row["part_id"], fgnr_row["sound_id"]
            )
            mixture, pure_fgnd = mixin(mixture, fgnd_file_name, snr)

        # If SNR is GTE 100, we enter a special mode with no background
        if self._forced_snr is not None and self._forced_snr >= 100:
            mixture = pure_fgnd

        # Conversion to STFT
        complex_mix = librosa.core.stft(mixture, n_fft=N_FFT, hop_length=HOP_LENGTH)[
            : N_FFT // 2, : TRAINING_LENGTH * FRAMES_PER_SECOND
        ]
        mix_real = np.real(complex_mix) / STFT_SCALE
        mix_imag = np.imag(complex_mix) / STFT_SCALE
        mix = np.concatenate((mix_real[:, :, None], mix_imag[:, :, None]), axis=-1)

        complex_fgnd = librosa.core.stft(pure_fgnd, n_fft=N_FFT, hop_length=HOP_LENGTH)[
            : N_FFT // 2, : TRAINING_LENGTH * FRAMES_PER_SECOND
        ]
        fgnd_real = np.real(complex_fgnd) / STFT_SCALE
        fgnd_imag = np.imag(complex_fgnd) / STFT_SCALE
        cirm_real, cirm_imag = generate_mask(fgnd_real, fgnd_imag, mix_real, mix_imag)

        fgnd_real_after_mask, fgnd_imag_after_mask = apply_mask(
            cirm_real, cirm_imag, mix_real, mix_imag
        )
        fgnd_real_after_mask *= STFT_SCALE
        fgnd_imag_after_mask *= STFT_SCALE
        complex_fgnd_after_mask = fgnd_real_after_mask + 1j * fgnd_imag_after_mask
        mask, margin = self.get_binary_mask_and_margin(
            pure_fgnd, complex_fgnd_after_mask
        )

        if VISUALIZE_DATASET:
            visualize_mix(
                complex_mix, complex_fgnd_after_mask, mask, margin, HOP_LENGTH, SAMPLE_RATE
            )

        weights = 1.0 - margin
        outs = np.concatenate(
            (
                mask[:, :, None],
                weights[:, :, None],
                cirm_real[:, :, None],
                cirm_imag[:, :, None],
            ),
            axis=-1,
        )

        return mix, outs, fgnd_ptr, (pure_fgnd != 0.0).any()

    def get_val_data(self):
        return self.x_val, self.y_val

    def get_test_data(self):
        return self.x_test, self.y_test

    def get_binary_mask_and_margin(self, wave, spectrogram):
        if (wave == 0.0).all():
            return np.zeros(spectrogram.shape, dtype=np.float32), np.zeros(
                spectrogram.shape, dtype=np.float32
            )

        S, phase = librosa.magphase(spectrogram)
        db = librosa.amplitude_to_db(S, amin=1e-10, top_db=150.0)

        threshold = db.max() - 60.0
        mask = np.zeros(db.shape, dtype=np.float32)
        mask[np.where(db > threshold)] = 1.0

        struct = ndimage.generate_binary_structure(2, 2)
        mask = ndimage.binary_dilation(mask, structure=struct, iterations=2).astype(
            mask.dtype
        )
        margin = ndimage.binary_dilation(mask, structure=struct, iterations=4).astype(
            mask.dtype
        )
        margin -= mask

        return mask, margin
