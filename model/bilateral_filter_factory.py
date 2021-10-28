'''
High-dimensional filter implemented in TF 2.x
Spatial high-dim filter if `features` is `None`
Bilateral high-dim filter otherwise
'''

import numpy as np
import tensorflow as tf

def clamp(min_value: tf.float32, max_value: tf.float32, x: tf.Tensor) -> tf.Tensor:
    return tf.maximum(min_value, tf.minimum(max_value, x))

# Method to get left and right indices of slice interpolation
def get_both_indices(size: tf.int32, coord: tf.Tensor) -> tf.Tensor:
    left_index = clamp(0, size - 1, tf.cast(coord, tf.int32))
    right_index = clamp(0, size - 1, left_index + 1)
    return left_index, right_index

@tf.function
def bilateral_high_dim_filter(inp: tf.Tensor, features: tf.Tensor, height: int, width: int, space_sigma: float=16, range_sigma: float=.25, padding_xy: int=2, padding_z: int=2) -> None:
    # For flattening
    size = height * width 
    # Bilateral convolutional dimension
    dim = 5
    # Number of iteration for convn
    n_iter = 2

    # Initialize a spatial high-dim filter if `features` is None; otherwise a bilateral one and `features` should be three-channel and channel-last
    tf.debugging.assert_equal(tf.shape(features)[-1], 3)
    tf.debugging.assert_equal(tf.rank(features), 3)

    # Height and width of data grid, scala, dtype int
    small_height = tf.cast(tf.cast(height - 1, tf.float32) / space_sigma, dtype=tf.int32) + 1 + 2 * padding_xy
    small_width = tf.cast(tf.cast(width - 1, tf.float32) / space_sigma, dtype=tf.int32) + 1 + 2 * padding_xy

    # Space coordinates, shape (h, w), dtype int
    yy, xx = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij') # (h, w)
    yy, xx = tf.cast(yy, tf.float32), tf.cast(xx, tf.float32)
    # Spatial coordinates of splat, shape (h, w)
    splat_yy = tf.cast(yy / space_sigma + .5, tf.int32) + padding_xy
    splat_xx = tf.cast(xx / space_sigma + .5, tf.int32) + padding_xy
    # Spatial coordinates of slice, shape (h, w)
    slice_yy = tf.cast(yy, tf.float32) / space_sigma + padding_xy
    slice_xx = tf.cast(xx, tf.float32) / space_sigma + padding_xy

    # Spatial interpolation index of slice
    y_index, yy_index = get_both_indices(small_height, slice_yy) # (h, w)
    x_index, xx_index = get_both_indices(small_width, slice_xx) # (h, w)

    # Spatial interpolation factor of slice
    y_alpha = tf.reshape(slice_yy - tf.cast(y_index, tf.float32), [-1, ]) # (h x w, )
    x_alpha = tf.reshape(slice_xx - tf.cast(x_index, tf.float32), [-1, ]) # (h x w, )

    # Decompose `features` into r, g, and b channels
    r, g, b = features[..., 0], features[..., 1], features[..., 2]
    r_min, r_max = tf.reduce_min(r), tf.reduce_max(r)
    g_min, g_max = tf.reduce_min(g), tf.reduce_max(g)
    b_min, b_max = tf.reduce_min(b), tf.reduce_max(b)
    r_delta, g_delta, b_delta = r_max - r_min, g_max - g_min, b_max - b_min
    # Range coordinates, shape (h, w), dtype float
    rr, gg, bb = r - r_min, g - g_min, b - b_min

    # Depths of data grid
    small_rdepth = tf.cast(r_delta / range_sigma, tf.int32) + 1 + 2 * padding_z
    small_gdepth = tf.cast(g_delta / range_sigma, tf.int32) + 1 + 2 * padding_z
    small_bdepth = tf.cast(b_delta / range_sigma, tf.int32) + 1 + 2 * padding_z

    # Range coordinates, shape (h, w)
    splat_rr = tf.cast(rr / range_sigma + .5, tf.int32) + padding_z
    splat_gg = tf.cast(gg / range_sigma + .5, tf.int32) + padding_z
    splat_bb = tf.cast(bb / range_sigma + .5, tf.int32) + padding_z

    # Range coordinates, shape (h, w)
    slice_rr = rr / range_sigma + padding_z
    slice_gg = gg / range_sigma + padding_z
    slice_bb = bb / range_sigma + padding_z

    # Slice interpolation range coordinate pairs
    r_index, rr_index = get_both_indices(small_rdepth, slice_rr) # (h, w)
    g_index, gg_index = get_both_indices(small_gdepth, slice_gg) # (h, w)
    b_index, bb_index = get_both_indices(small_bdepth, slice_bb) # (h, w)

    # Interpolation factors
    r_alpha = tf.reshape(slice_rr - tf.cast(r_index, tf.float32), [-1, ]) # (h x w, ) 
    g_alpha = tf.reshape(slice_gg - tf.cast(g_index, tf.float32), [-1, ]) # (h x w, )
    b_alpha = tf.reshape(slice_bb - tf.cast(b_index, tf.float32), [-1, ]) # (h x w, )

    # Bilateral interpolation index and factor
    interp_indices = [y_index, yy_index, x_index, xx_index, r_index, rr_index, g_index, gg_index, b_index, bb_index] # (10, h x w)
    alphas = [1. - y_alpha, y_alpha, 1. - x_alpha, x_alpha, 1. - r_alpha, r_alpha, 1. - g_alpha, g_alpha, 1. - b_alpha, b_alpha] # (10, h x w)

    # Method of coordinate transformation
    def coord_transform(idx):
        return tf.reshape((((idx[:, 0, :] * small_width + idx[:, 1, :]) * small_rdepth + idx[:, 2, :]) * small_gdepth + idx[:, 3, :]) * small_bdepth + idx[:, 4, :], [-1, ]) # (2^dim x h x w, )

    # Initialize interpolation
    offset = tf.range(dim) * 2 # [dim, ]
    # Permutation
    permutations = tf.stack(tf.meshgrid(tf.range(2), tf.range(2), tf.range(2), tf.range(2), tf.range(2), indexing='ij'), axis=-1)
    permutations = tf.reshape(permutations, [-1, dim]) # [2^dim, dim]
    permutations += offset[tf.newaxis, ...]
    permutations = tf.reshape(permutations, [-1, ]) # Flatten, [2^dim x dim]
    alpha_prods = tf.reshape(tf.gather(alphas, permutations), [-1, dim, size]) # [2^dim, dim, h x w]
    idx = tf.reshape(tf.gather(interp_indices, permutations), [-1, dim, size]) # [2^dim, dim, h x w]

    # Shape of bilateral data grid
    data_shape = [small_height, small_width, small_rdepth, small_gdepth, small_bdepth]
    data_size = small_height * small_width * small_rdepth * small_gdepth * small_bdepth

    # Bilateral splat coordinates, shape (h x w, )
    splat_coords = (((splat_yy * small_width + splat_xx) * small_rdepth + splat_rr) * small_gdepth + splat_gg) * small_bdepth + splat_bb
    splat_coords = tf.reshape(splat_coords, [-1, ]) # (h x w, )

    # Interpolation indices and alphas of bilateral slice
    slice_idx = coord_transform(idx)
    alpha_prod = tf.math.reduce_prod(alpha_prods, axis=1)

    # `inp` should be 2-dim because computed in a channel-wise way
    tf.debugging.assert_equal(tf.rank(inp), 3)
    # Channel-last to channel-first because tf.map_fn
    inpT = tf.transpose(inp, (2, 0, 1))


    def ch_filter(inp_ch: tf.Tensor) -> tf.Tensor:
        # Filter each channel
        inp_flat = tf.reshape(inp_ch, [-1, ]) # (h x w)
        # ==== Splat ====
        data_flat = tf.math.bincount(splat_coords, weights=inp_flat, minlength=data_size, maxlength=data_size, dtype=tf.float32)
        data = tf.reshape(data_flat, data_shape)
        
        # ==== Blur ====
        buffer = tf.zeros_like(data)
        perm = [1, 2, 3, 4, 0]

        for _ in range(n_iter):
            buffer, data = data, buffer

            for _ in range(dim):
                newdata = (buffer[:-2] + buffer[2:]) / 2.
                data = tf.concat([data[:1], newdata, data[-1:]], axis=0)
                data = tf.transpose(data, perm=perm)
                buffer = tf.transpose(buffer, perm=perm)

        del buffer

        # ==== Slice ====
        data_slice = tf.gather(tf.reshape(data, [-1, ]), slice_idx) # (2^dim x h x w)
        data_slice = tf.reshape(data_slice, [-1, size]) # (2^dim, h x w)
        interpolations = alpha_prod * data_slice
        interpolation = tf.reduce_sum(interpolations, axis=0)
        interpolation = tf.reshape(interpolation, [height, width])

        return interpolation

    outT = tf.map_fn(ch_filter, inpT)
    out = tf.transpose(outT, (1, 2, 0)) # Channel-first to channel-last
    
    return out
