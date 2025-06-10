import os

import random
import time

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

BGND_REC_LENGTH = 10
TRAINING_LENGTH = 8
SAMPLE_RATE = 8000

RMS_FRAME_LEN = 2000
RMS_EPSILON = 0.000001
RMS_MINIMUM = -60.0

FGND_MIN_LENGTH = 1000


def read_wave(filename):
    while True:
        try:
            wave, sample_rate = sf.read(filename)
            break
        except:
            time.sleep(60)  # Minute
    return wave, sample_rate


def write_wave(filename, wave, sample_rate):
    sf.write(filename, wave, sample_rate)


def get_background(filename):
    wave, sample_rate = read_wave(filename)
    assert wave.shape[0] == BGND_REC_LENGTH * SAMPLE_RATE
    assert sample_rate == SAMPLE_RATE

    start_sample = random.randint(0, wave.shape[0] - TRAINING_LENGTH * SAMPLE_RATE)
    wave = wave[start_sample : start_sample + TRAINING_LENGTH * SAMPLE_RATE]

    wave_rms = np.sqrt(np.mean(wave ** 2.0))

    bgnd_lvl_min = -50
    bgnd_lvl_max = -20
    target_level = random.random() * (bgnd_lvl_min - bgnd_lvl_max) + bgnd_lvl_max
    gain = 10.0 ** (target_level / 20.0) / wave_rms

    if np.isfinite(gain):
        wave *= gain

    return wave


def get_fgnd_small_mask(wave):
    rms = librosa.feature.rms(wave, frame_length=RMS_FRAME_LEN, hop_length=80)
    rms = 20.0 * np.log10(rms + RMS_EPSILON)
    rms = (rms[0] >= RMS_MINIMUM) * 1.0
    rms = np.repeat(rms, 80)
    return rms[: wave.shape[0]]


def get_random_wave_start(bgnd_len, wave):
    fgnd_len = wave.shape[0]
    if random.choice([0, 1, 1, 1, 1, 1]):
        return wave, random.randint(0, bgnd_len - fgnd_len)
    else:
        cut = random.randint(1, fgnd_len - FGND_MIN_LENGTH)
        if random.choice([0, 1]):
            return wave[cut:], 0
        else:
            return wave[:cut], bgnd_len - cut


def mixin(bgnd, filename, forced_snr=None):
    wave, sample_rate = read_wave(filename)
    assert wave.shape[0] < TRAINING_LENGTH * SAMPLE_RATE
    assert sample_rate == SAMPLE_RATE

    wave = np.trim_zeros(wave)
    assert wave.shape[0] > FGND_MIN_LENGTH

    wave, start = get_random_wave_start(bgnd.shape[0], wave)

    bgnd_part = bgnd[start : start + wave.shape[0]]
    bgnd_part_rms = np.sqrt(np.mean(bgnd_part ** 2.0))
    bgnd_part_db = 20.0 * np.log10(bgnd_part_rms + RMS_EPSILON)

    if forced_snr is not None:
        target_wave_db = bgnd_part_db + forced_snr
    else:
        target_wave_db = bgnd_part_db + random.choice([-10, -5, 0, 5])

    wave_rms = np.sqrt(np.mean(wave ** 2.0))
    gain_wave = 10.0 ** (target_wave_db / 20.0) / wave_rms

    if np.isfinite(gain_wave):
        wave *= gain_wave
    bgnd[start : start + wave.shape[0]] += wave

    pure_wave = np.zeros_like(bgnd)
    pure_wave[start : start + wave.shape[0]] += wave

    return bgnd, pure_wave
