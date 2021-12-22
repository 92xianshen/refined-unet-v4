# -*- coding: utf-8 -*-

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from create_model import create_refined_unet_v4

theta_alpha, theta_beta, theta_gamma = 80, .0625, 3

CROP_HEIGHT, CROP_WIDTH = 512, 512
num_bands = 3
num_classes = 4
batch_size = 1
unet_path = 'pretrained/'
data_path = '../../datasets/rice/test_set/'
output_path = '../../rice_result/'

model = create_refined_unet_v4(input_channels=num_bands,
                               num_classes=num_classes,
                               theta_alpha=theta_alpha,
                               theta_beta=theta_beta,
                               theta_gamma=theta_gamma,
                               spatial_compat=3.0,
                               bilateral_compat=10.0,
                               num_iterations=10,
                               gt_prob=0.7,
                               unet_pretrained=unet_path)

def load_testset(filenames, batch_size=batch_size):
    """ Load a tensorflow TFDataset file as a test set
    """
    test_dataset = tf.data.TFRecordDataset(filenames)

    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string), 
        'height': tf.io.FixedLenFeature([], tf.int64), 
        'width': tf.io.FixedLenFeature([], tf.int64), 
        'n_channels': tf.io.FixedLenFeature([], tf.int64), 
    }

    def _parse_function(example_proto):
        example = tf.io.parse_single_example(
            example_proto, feature_description)

        x = tf.io.decode_raw(example['x'], tf.uint8)
        y = tf.io.decode_raw(example['y'], tf.int32)
        height, width, n_channels = example['height'], example['width'], example['n_channels']

        x = tf.reshape(x, [height, width, n_channels])
        y = tf.reshape(y, [height, width])

        x = tf.cast(x, tf.float32)
        x = (x - tf.reduce_min(x)) / \
            (tf.reduce_max(x) - tf.reduce_min(x) + 1e-10)
        y = tf.cast(y, tf.int32)

        example['x'] = x
        example['y'] = y

        return example

    test_dataset = test_dataset.map(_parse_function).batch(
        batch_size, drop_remainder=True)

    return test_dataset

@tf.function
def inference(x):
    logits, rfn_logits = model(x, training=False)
    preds, rfns = tf.math.argmax(logits, axis=-1), tf.math.argmax(rfn_logits, axis=-1)
    return preds, rfns


def main():
    test_names = os.listdir(data_path)
    test_fullnames = [os.path.join(data_path, name) for name in test_names]
    test_set = load_testset(test_fullnames, batch_size=batch_size)

    # ->> output folders
    if not os.path.exists(os.path.join(output_path, 'rfn')):
        os.makedirs(os.path.join(output_path, 'rfn', 'png'))
        os.makedirs(os.path.join(output_path, 'rfn', 'npz'))
    if not os.path.exists(os.path.join(output_path, 'pred')):
        os.makedirs(os.path.join(output_path, 'pred', 'png'))
        os.makedirs(os.path.join(output_path, 'pred', 'npz'))
    if not os.path.exists(os.path.join(output_path, 'ref')):
        os.makedirs(os.path.join(output_path, 'ref', 'png'))
        os.makedirs(os.path.join(output_path, 'ref', 'npz'))

    for record, name in zip(test_set, test_names):
        rfn_npz_name = os.path.join(output_path, 'rfn', 'npz', name.replace('.tfrecord', '.npz'))
        rfn_png_name = os.path.join(output_path, 'rfn', 'png', name.replace('.tfrecord', '.png'))
        pred_npz_name = os.path.join(output_path, 'pred', 'npz', name.replace('.tfrecord', '.npz'))
        pred_png_name = os.path.join(output_path, 'pred', 'png', name.replace('.tfrecord', '.png'))
        ref_npz_name = os.path.join(output_path, 'ref', 'npz', name.replace('.tfrecord', '.npz'))
        ref_png_name = os.path.join(output_path, 'ref', 'png', name.replace('.tfrecord', '.png'))

        x, y = record['x'], record['y']

        preds, rfns = inference(x)
        pred, rfn = preds.numpy()[0], rfns.numpy()[0]
        ref = y.numpy()[0]

        # ->> save as npz
        np.savez(pred_npz_name, pred)
        np.savez(rfn_npz_name, rfn)
        np.savez(ref_npz_name, ref)

        # ->> save as png
        plt.imsave(pred_png_name, pred, cmap='gray')
        plt.imsave(rfn_png_name, rfn, cmap='gray')
        plt.imsave(ref_png_name, ref, cmap='gray')

if __name__ == '__main__':
    main()