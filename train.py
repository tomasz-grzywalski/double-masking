import os
import gc

import random
import numpy as np
from math import ceil

import tensorflow as tf

from utils.dataset import Dataset, N_FFT, FRAMES_PER_SECOND
from utils.tf_utils import LearningRateScheduler, ComboLoss, MaskLoss, CirmLoss
from utils.recordings import TRAINING_LENGTH

from models.unet import get_model

RANDOM_SEED = 119

DATASET_ROOT = "#TODO"
OUTPUT_ROOT = "#TODO"
SNAPSHOT_FILE_NAME = "best_snapshot.ckpt"

LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.97

EPOCHS = 100

BATCH_SIZE = 12


def generate_evaltest_data(x, y):
    for idx in range(ceil(x.shape[0] / BATCH_SIZE)):
        yield x[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE], y[
            idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE
        ]


def main():
    dataset = Dataset(DATASET_ROOT)

    lr_decay_callback = LearningRateScheduler(
        learning_rate=LEARNING_RATE, decay=LEARNING_RATE_DECAY
    )

    model = get_model(N_FFT // 2, TRAINING_LENGTH * FRAMES_PER_SECOND, 2)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    model.compile(
        optimizer=optimizer, loss=ComboLoss(), metrics=[MaskLoss(), CirmLoss()]
    )

    x_val, y_val = dataset.get_val_data()
    x_test, y_test = dataset.get_test_data()

    best_loss = float("inf")
    weights_path = os.path.join(OUTPUT_ROOT, SNAPSHOT_FILE_NAME)
    for epoch in range(EPOCHS):
        print("")
        print(f"# Epoch {epoch + 1}")
        optimizer.learning_rate.assign(LEARNING_RATE * (LEARNING_RATE_DECAY ** epoch))
        reset_metrics = True
        for x_train, y_train in dataset.iterate_train_samples(BATCH_SIZE):
            loss = model.train_on_batch(x_train, y_train, reset_metrics=reset_metrics)
            reset_metrics = False

        print(f"# Epoch {epoch + 1} loss: {loss}")

        val_loss, val_mask_loss, val_cirm_loss = model.evaluate(
            generate_evaltest_data(x_val, y_val), verbose=2
        )

        if val_loss < best_loss:
            print(
                f" * New best loss!: {val_loss} Mask loss: {val_mask_loss} CIRM loss: {val_cirm_loss}"
            )
            best_loss = val_loss
            model.save_weights(weights_path)

        gc.collect()
        tf.keras.backend.clear_session()

    model.load_weights(weights_path)

    print("")
    print(f"# Testing best model")
    test_loss, test_mask_loss, test_cirm_loss = model.evaluate(
        generate_evaltest_data(x_test, y_test), verbose=2
    )
    print(
        f"Test loss!: {test_loss} Mask loss: {test_mask_loss} CIRM loss: {test_cirm_loss}"
    )


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    main()
