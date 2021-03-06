"""
MIT License

Copyright (c) 2021 Libin Jiao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os

import numpy as np
import tensorflow as tf

from model.CRFLayer import CRFLayer
from model.UNet import UNet


def create_refined_unet_v4(input_channels, num_classes, theta_alpha=80, theta_beta=.0625, theta_gamma=3.0, spatial_compat=3.0, bilateral_compat=10.0, num_iterations=10, gt_prob=0.7, unet_pretrained=None):
    """ Create Refined UNet v2 """

    # Input
    inputs = tf.keras.Input(
        shape=[None, None, input_channels], name='inputs')

    # Create UNet
    unet = UNet()

    # Restore pretrained UNet
    if unet_pretrained:
        checkpoint = tf.train.Checkpoint(model=unet)
        checkpoint.restore(tf.train.latest_checkpoint(unet_pretrained))
        print('Checkpoint restored, at {}'.format(
            tf.train.latest_checkpoint(unet_pretrained)))

    # Create CRF layer
    crf = CRFLayer(num_classes, theta_alpha, theta_beta, theta_gamma,
                   spatial_compat, bilateral_compat, num_iterations)

    # RGB channels, scale [0, 1]
    image = inputs[..., 4:1:-1]

    # Only forward
    logits = unet(inputs)
    probs = tf.nn.softmax(logits, name='logits2probs')
    unary = -tf.math.log(probs * gt_prob, name='probs2unary')

    refined_logits = crf(unary=unary[0], image=image[0])
    refined_logits = refined_logits[tf.newaxis, ...]

    return tf.keras.Model(inputs=inputs, outputs=[logits, refined_logits])
