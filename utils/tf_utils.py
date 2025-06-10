import math

import numpy as np
import tensorflow as tf


class LearningRateScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, *args, learning_rate, decay, **kwargs):
        self._learning_rate = learning_rate
        self._decay_factor = decay
        super().__init__(self._implementation, verbose=0, *args, **kwargs)

    def _implementation(self, epoch):
        return self._learning_rate * math.pow(self._decay_factor, epoch)


class ComboLoss(tf.keras.losses.Loss):
    def __init__(self, name="combo_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        gt_mask = y_true[:, :, :, 0:1]
        gt_mask_weights = y_true[:, :, :, 1:2]
        gt_cirm = y_true[:, :, :, 2:]

        pred_mask = y_pred[:, :, :, 0:1]
        pred_mask = tf.math.sigmoid(pred_mask)
        pred_cirm = y_pred[:, :, :, 1:]

        mask_error = tf.square(gt_mask - pred_mask) * gt_mask_weights
        mask_error = tf.math.reduce_sum(mask_error) / tf.math.reduce_sum(
            gt_mask_weights
        )

        # Optionally use disconnected mask instead of GT mask
        # disconnected_mask = tf.stop_gradient(tf.cast(tf.math.greater(pred_mask, 0.5), dtype=tf.float32))

        cirm_error = tf.square(gt_cirm - pred_cirm) * gt_mask
        cirm_error = tf.math.reduce_sum(cirm_error) / (
            tf.math.reduce_sum(gt_mask) + tf.keras.backend.epsilon()
        )

        return mask_error + cirm_error


class MaskLoss(tf.keras.metrics.Metric):
    def __init__(self, name="mask_loss", **kwargs):
        super(MaskLoss, self).__init__(name=name, **kwargs)
        self.mask_loss_sum = self.add_weight(name="mask_loss_sum", initializer="zeros")
        self.mask_loss_cnt = self.add_weight(name="mask_loss_cnt", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_mask = y_true[:, :, :, 0:1]
        gt_mask_weights = y_true[:, :, :, 1:2]

        pred_mask = y_pred[:, :, :, 0:1]
        pred_mask = tf.math.sigmoid(pred_mask)

        mask_error = tf.square(gt_mask - pred_mask) * gt_mask_weights
        mask_error = tf.math.reduce_sum(mask_error) / tf.math.reduce_sum(
            gt_mask_weights
        )

        self.mask_loss_sum.assign_add(mask_error)
        self.mask_loss_cnt.assign_add(1)

    def result(self):
        return self.mask_loss_sum / self.mask_loss_cnt

    def reset_state(self):
        self.mask_loss_sum.assign(0.0)
        self.mask_loss_cnt.assign(0.0)


class CirmLoss(tf.keras.metrics.Metric):
    def __init__(self, name="cirm_loss", **kwargs):
        super(CirmLoss, self).__init__(name=name, **kwargs)
        self.cirm_loss_sum = self.add_weight(name="cirm_loss_sum", initializer="zeros")
        self.cirm_loss_cnt = self.add_weight(name="cirm_loss_cnt", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        gt_mask = y_true[:, :, :, 0:1]
        gt_cirm = y_true[:, :, :, 2:]

        pred_cirm = y_pred[:, :, :, 1:]

        # Optionally use disconnected mask instead of GT mask
        # disconnected_mask = tf.stop_gradient(tf.cast(tf.math.greater(pred_mask, 0.5), dtype=tf.float32))

        cirm_error = tf.square(gt_cirm - pred_cirm) * gt_mask
        cirm_error = tf.math.reduce_sum(cirm_error) / (
            tf.math.reduce_sum(gt_mask) + tf.keras.backend.epsilon()
        )

        self.cirm_loss_sum.assign_add(cirm_error)
        self.cirm_loss_cnt.assign_add(1)

    def result(self):
        return self.cirm_loss_sum / self.cirm_loss_cnt

    def reset_state(self):
        self.cirm_loss_sum.assign(0.0)
        self.cirm_loss_cnt.assign(0.0)
