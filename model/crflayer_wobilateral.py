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

import sys
import numpy as np
import tensorflow as tf
from model.spatial_filter_factory import spatial_high_dim_filter

# from spatial_filter_factory import spatial_high_dim_filter
# from filter_factory.bilateral_filter_factory import bilateral_high_dim_filter


def _diagonal_compatibility(shape):
    return tf.eye(shape[0], shape[1], dtype=np.float32)


def _potts_compatibility(shape):
    return -1 * _diagonal_compatibility(shape)


class CRFLayer(tf.keras.layers.Layer):
    """ A layer implementing CRF """

    def __init__(self, num_classes, theta_gamma, spatial_compat, num_iterations):
        super(CRFLayer, self).__init__()
        self.num_classes = num_classes
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations

        self.spatial_weights = spatial_compat * _diagonal_compatibility((num_classes, num_classes))
        self.compatibility_matrix = _potts_compatibility((num_classes, num_classes))

    def call(self, unary, image):
        """
        The order of parameters: I, p
        """
        assert len(image.shape) == 3 and len(unary.shape) == 3

        unary_shape = tf.shape(unary)
        height, width = unary_shape[0], unary_shape[1]
        
        all_ones = tf.ones([height, width, self.num_classes], dtype=tf.float32)
        
        # Compute symmetric weight
        spatial_norm_vals = spatial_high_dim_filter(all_ones, height, width, space_sigma=self.theta_gamma)
        spatial_norm_vals = 1. / (spatial_norm_vals ** .5 + 1e-20)

        # Initialize Q
        Q = tf.nn.softmax(-unary)

        for i in range(self.num_iterations):
            tmp1 = -unary

            # Symmetric normalization and spatial message passing
            spatial_out = spatial_high_dim_filter(Q * spatial_norm_vals, height, width, space_sigma=self.theta_gamma)
            spatial_out *= spatial_norm_vals

            # Message passing
            spatial_out = tf.reshape(spatial_out, [-1, self.num_classes])
            spatial_out = tf.matmul(spatial_out, self.spatial_weights)
            message_passing = spatial_out

            # Compatibility transform
            pairwise = tf.matmul(message_passing, self.compatibility_matrix)
            pairwise = tf.reshape(pairwise, unary_shape)

            # Local update
            tmp1 -= pairwise

            # Normalize
            Q = tf.nn.softmax(tmp1)

        return Q