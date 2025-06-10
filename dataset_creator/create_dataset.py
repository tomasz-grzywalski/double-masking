import os
import random

import numpy as np
import soundfile as sf
import librosa
import librosa.display

import noisereduce as nr
from pedalboard import Pedalboard, HighpassFilter

from model.stride_three import get_model

import matplotlib.pyplot as plt

FFBF_ROOT = "#TDOD"
MAX_FFBF_ID = 51598
MAX_FFBF_N = 5

SAMPLE_RATE = 8000
HOP_LENGTH = 80
REC_LENGTH = 10

RMS_FRAME_LEN = 2000
RMS_EPSILON = 0.000001
RMS_MINIMUM = -120.0

MODEL_NUM_CLASSES = 515
MODEL_NUM_SAMPLES = 3 * SAMPLE_RATE
MODEL_WEIGHTS = "./dataset_creator/model/best_snapshot.ckpt"

BGND_MIN_ENERGY = -60.0
BGND_MAX_EMBEDDING_DIST = 7.5

FGND_MIN_TOP_ENERGY = -35.0
FGND_MIN_SAMPLE_DYNAMIC = 35.0
FGND_MAX_LEN = [int(max_len * SAMPLE_RATE // HOP_LENGTH) for max_len in [1, 2, 3, 4]]
FGND_FADE_IN_OUT_LEN = 20 * HOP_LENGTH
FGND_MAX_SOUNDS = 5

RANDOMIZE_ORDER = False

VISUALIZE = False
VISUALIZE_BGND_DIR = "./Snippet/Background/"
VISUALIZE_FGND_DIR = "./Snippet/Foreground/"

SAVE_DATASET = False
DESTINATION_BGND = "./Output/Background/"
DESTINATION_FGND = "./Output/Foreground/"


def limit_gpu_memory():
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2048)]
            )
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)


def get_rms(wave):
    rms = librosa.feature.rms(wave, frame_length=RMS_FRAME_LEN, hop_length=HOP_LENGTH)
    rms = 20.0 * np.log10(rms + RMS_EPSILON)
    return rms[0]


def is_background(wave, rms, model):
    preds = model(wave[None, :, None]).numpy()[0]

    mean_preds = preds.mean(axis=0)
    dsts = [np.mean(np.power(pred - mean_preds, 2)) for pred in preds]
    dst = np.max(dsts)

    return (
        np.all(rms > BGND_MIN_ENERGY) and dst <= BGND_MAX_EMBEDDING_DIST,
        f"{dst:.03f}",
    )


def get_fgnd_sounds(rms):
    if np.all(rms < FGND_MIN_TOP_ENERGY):
        return []

    fgnd_sounds = list()
    mask = rms[:-1].copy()
    for max_len in FGND_MAX_LEN:
        fgnd_sounds, mask = find_fgnd_sounds_of_size(fgnd_sounds, mask, max_len)

    return fgnd_sounds


def find_fgnd_sounds_of_size(fgnd_sounds, mask, size):
    work = mask.copy()

    while np.any(np.isfinite(work)):
        if len(fgnd_sounds) >= FGND_MAX_SOUNDS:
            break

        idx_of_max = np.nanargmax(work)
        if work[idx_of_max] < FGND_MIN_TOP_ENERGY:
            break

        analysis_min = max(idx_of_max - size // 2, 0)
        if np.all(np.isnan(work[analysis_min:idx_of_max])):
            work[idx_of_max] = np.nan
            continue
        idx_of_left_min = np.nanargmin(work[analysis_min:idx_of_max]) + analysis_min

        analysis_max = min(idx_of_max + size // 2 + 1, work.shape[0])
        if np.all(np.isnan(work[idx_of_max + 1 : analysis_max])):
            work[idx_of_max] = np.nan
            continue
        idx_of_right_min = (
            np.nanargmin(work[idx_of_max + 1 : analysis_max]) + idx_of_max + 1
        )

        assert idx_of_right_min - idx_of_left_min <= size
        if (
            work[idx_of_max] - work[idx_of_left_min] < FGND_MIN_SAMPLE_DYNAMIC
            and work[idx_of_max] - work[idx_of_right_min] < FGND_MIN_SAMPLE_DYNAMIC
        ):
            work[idx_of_left_min : idx_of_right_min + 1] = np.nan
            continue

        if work[idx_of_left_min] < work[idx_of_right_min]:
            analysis_max = min(
                idx_of_max + (size - idx_of_max + idx_of_left_min) + 1, work.shape[0]
            )
            idx_of_right_min = (
                np.nanargmin(work[idx_of_max + 1 : analysis_max]) + idx_of_max + 1
            )
        else:
            analysis_min = max(idx_of_max - (size - idx_of_right_min + idx_of_max), 0)
            idx_of_left_min = np.nanargmin(work[analysis_min:idx_of_max]) + analysis_min

        if (
            work[idx_of_max] - work[idx_of_left_min] < FGND_MIN_SAMPLE_DYNAMIC
            or work[idx_of_max] - work[idx_of_right_min] < FGND_MIN_SAMPLE_DYNAMIC
            or np.any(np.isnan(work[idx_of_left_min : idx_of_right_min + 1]))
        ):
            work[idx_of_left_min : idx_of_right_min + 1] = np.nan
            continue

        work[idx_of_left_min : idx_of_right_min + 1] = np.nan
        mask[idx_of_left_min : idx_of_right_min + 1] = np.nan
        fgnd_sounds.append((idx_of_left_min, idx_of_right_min))

    return fgnd_sounds, mask


def extract_sounds(wave, sounds_beg_ends):
    sounds = list()

    for beg, end in sounds_beg_ends:
        sound = wave[beg * HOP_LENGTH : end * HOP_LENGTH]

        fade_in_out = np.ones_like(sound)
        fade_in_out[:FGND_FADE_IN_OUT_LEN] = np.arange(
            0.0, 1.0, 1.0 / FGND_FADE_IN_OUT_LEN
        )
        fade_in_out[-FGND_FADE_IN_OUT_LEN:] = np.arange(
            1.0, 0.0, -1.0 / FGND_FADE_IN_OUT_LEN
        )

        sound *= fade_in_out
        sounds.append(sound)

    return sounds


def add_spectrogram_to_vis(wave, axis, title):
    S, phase = librosa.magphase(librosa.stft(wave, hop_length=HOP_LENGTH, n_fft=768))
    img = librosa.display.specshow(
        librosa.amplitude_to_db(S, ref=np.max),
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        y_axis="linear",
        x_axis="time",
        ax=axis,
    )
    axis.set(title=title)


def visualize_sample(wave, rms, info, fgnd_sounds, sample_id, alt_wave=None):
    if not VISUALIZE:
        return

    rows = 2 if alt_wave is None else 3
    heigth = 15 if alt_wave is None else 21
    fig, ax = plt.subplots(nrows=rows, sharex=True, figsize=(35, heigth))
    times = librosa.times_like(rms, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
    ax[0].plot(times, rms, label="Energy")
    for beg, end in fgnd_sounds:
        snd_raster = np.ones_like(rms) * RMS_MINIMUM
        snd_raster[beg : end + 1] = rms[beg : end + 1].max()
        ax[0].plot(times, snd_raster, label="Sound")
    ax[0].set_ylim([-100.0, 0.0])
    title = f"Sample {sample_id}: {info}\nEnergy: Min: {rms.min():.2f} dB "
    title += f"Max: {rms.max():.2f} dB Mean: {rms.mean():.2f} dB"
    ax[0].set(title=title)
    ax[0].set(xticks=[])
    ax[0].set(ylabel="dB")
    ax[0].legend()
    ax[0].label_outer()

    add_spectrogram_to_vis(wave, ax[1], "Power spectrogram")
    if alt_wave is not None:
        add_spectrogram_to_vis(
            alt_wave, ax[2], "Power spectrogram of the original recording"
        )

    root = VISUALIZE_FGND_DIR if fgnd_sounds else VISUALIZE_BGND_DIR
    plt.savefig(os.path.join(root, f"{sample_id:05d}.png"))
    sf.write(os.path.join(root, f"{sample_id:05d}.wav"), wave, SAMPLE_RATE)


def visualize_sounds(sounds, sample_id):
    if not VISUALIZE:
        return

    for sound_id, sound in enumerate(sounds):
        fig, ax = plt.subplots(
            nrows=1, figsize=(int(3.5 * sound.shape[0] / SAMPLE_RATE), 9)
        )

        S, phase = librosa.magphase(
            librosa.stft(sound, hop_length=HOP_LENGTH, n_fft=768)
        )
        img = librosa.display.specshow(
            librosa.amplitude_to_db(S, ref=np.max),
            sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            y_axis="linear",
            x_axis="time",
            ax=ax,
        )
        ax.set(title="Power spectrogram")

        plt.savefig(
            os.path.join(VISUALIZE_FGND_DIR, f"{sample_id:05d}_{sound_id:02d}.png")
        )
        sf.write(
            os.path.join(VISUALIZE_FGND_DIR, f"{sample_id:05d}_{sound_id:02d}.wav"),
            sound,
            SAMPLE_RATE,
        )


def process_sample(sample, model, hp_filter, sample_id):
    rms = get_rms(sample)

    fgnd_sounds = list()
    background, info = is_background(sample, rms, model)
    if background:
        info = f"BGND (max embedding euclidean distance deviation: {info})"
        visualize_sample(sample, rms, info, fgnd_sounds, sample_id)
    else:
        fgnd_sample = nr.reduce_noise(
            y=sample, sr=SAMPLE_RATE, n_fft=512, stationary=False
        )
        if np.any(np.isnan(fgnd_sample)):
            return background, fgnd_sounds
        fgnd_sample = hp_filter(fgnd_sample, SAMPLE_RATE)

        fgnd_rms = get_rms(fgnd_sample)
        fgnd_sounds = get_fgnd_sounds(fgnd_rms)

        if fgnd_sounds:
            info = f"FGND ({len(fgnd_sounds)} sounds): {fgnd_sounds}"
            visualize_sample(
                fgnd_sample, fgnd_rms, info, fgnd_sounds, sample_id, alt_wave=sample
            )

            fgnd_sounds = extract_sounds(fgnd_sample, fgnd_sounds)
            visualize_sounds(fgnd_sounds, sample_id)

    return background, fgnd_sounds


def save_file(wave, root, idx1, idx2, idx3=None):
    if not SAVE_DATASET:
        return

    subdir = os.path.join(root, f"{(idx1 // 100):04d}")

    if not os.path.isdir(subdir):
        os.mkdir(subdir)

    if idx3 is not None:
        filename = os.path.join(subdir, f"{idx1:05d}_{idx2}_{idx3:02d}.wav")
    else:
        filename = os.path.join(subdir, f"{idx1:05d}_{idx2}.wav")

    sf.write(filename, wave, SAMPLE_RATE)


def main():
    model = get_model(MODEL_NUM_SAMPLES, MODEL_NUM_CLASSES)
    model.load_weights(MODEL_WEIGHTS)

    hp_filter = Pedalboard([HighpassFilter(cutoff_frequency_hz=250)])

    num_b, num_f, num_n, num_s = 0, 0, 0, 0
    fgnd_lenghts = list()
    for original_rec_id in range(MAX_FFBF_ID + 1):
        if RANDOMIZE_ORDER:
            original_rec_id = random.randint(0, MAX_FFBF_ID)

        for original_part_id in range(1, MAX_FFBF_N + 1):
            dir_name = f"{(original_rec_id // 100):03d}"
            file_name = f"{original_rec_id:05d}_{original_part_id}.wav"
            full_file_name = os.path.join(FFBF_ROOT, dir_name, file_name)
            if not os.path.exists(full_file_name):
                continue

            sample, _ = librosa.load(full_file_name, SAMPLE_RATE)
            assert sample.shape == (REC_LENGTH * SAMPLE_RATE,)

            bgnd, fgnd_sounds = process_sample(
                sample, model, hp_filter, num_b + num_f + num_n
            )
            assert not (bgnd and fgnd_sounds)

            if bgnd:
                save_file(sample, DESTINATION_BGND, original_rec_id, original_part_id)
                num_b += 1
            elif fgnd_sounds:
                for fgnd_sound_id, fgnd_sound in enumerate(fgnd_sounds):
                    save_file(
                        fgnd_sound,
                        DESTINATION_FGND,
                        original_rec_id,
                        original_part_id,
                        fgnd_sound_id + 1,
                    )
                    num_s += 1
                    fgnd_lenghts.append(fgnd_sound.shape[0] / SAMPLE_RATE)
                num_f += 1
            else:
                num_n += 1

            if fgnd_lenghts:
                msg = f"Num BGND: {num_b} Num FGND: {num_f} Num None: {num_n} FGND snds: {num_s} "
                msg += (
                    f"ave len: {np.mean(fgnd_lenghts):.3f} min: {np.min(fgnd_lenghts)} "
                )
                msg += f"max: {np.max(fgnd_lenghts)}"
                print(msg)


if __name__ == "__main__":
    random.seed(126)
    limit_gpu_memory()
    main()
