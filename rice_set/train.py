# -*- coding: utf-8 -*-

""" 6-categorical Segmentation of Bangladesh, using weighted unet.
"""

import os
import pprint
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model.UNet import UNet

# ==== Dataset ====
train_path, val_path = '../../datasets/rice/train_set/', '../../datasets/rice/val_set/'

restore = 0
n_classes = 4
epochs = 100
init_lr = 1e-4
decay_step = 100
checkpoint_path = 'checkpoint'

EPSILON = 1e-8

# ==== Distributed training configuration ====
strategy = tf.distribute.MirroredStrategy()

batch_size_per_replica = 1
global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
buffer_size = global_batch_size * 64


def load_dataset(filenames):
    """ Load TFRecord files as dataset
    """
    dataset = tf.data.TFRecordDataset(filenames)

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

    dataset = dataset.map(_parse_function).batch(
        global_batch_size, drop_remainder=True)

    return dataset


def train_loop():
    # ==== Load dataset ====
    train_names = os.listdir(train_path)
    train_names = [os.path.join(train_path, name) for name in train_names]

    val_fnames = os.listdir(val_path)
    val_fnames = [os.path.join(val_path, name) for name in val_fnames]

    train_set = load_dataset(train_names)
    val_set = load_dataset(val_fnames)

    # ==== Log ====
    logdir = os.path.join('logs/')
    file_writer = tf.summary.create_file_writer(logdir + 'metrics')
    file_writer.set_as_default()

    # ==== Visualization ====
    if not os.path.exists('Visualization/'):
        os.makedirs('Visualization')

    # ==== 2020-09-17 Distributed training ====
    with strategy.scope():
        # ==== Define loss function and validation metrics ====
        loss_object = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        # ==== Commented Weighted loss ====
        def compute_weighted_loss(logits, labels):
            labels = tf.reshape(labels, [-1])
            count = tf.math.bincount(labels, minlength=4, maxlength=4)
            class_weights = tf.reduce_max(count) / count

            labels_onehot = tf.one_hot(labels, depth=n_classes)
            logits = tf.reshape(logits, [-1, n_classes])
            weights = tf.gather(class_weights, labels)

            per_example_loss = loss_object(y_true=labels_onehot,
                               y_pred=logits, sample_weight=weights)

            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        # ==== End ====

        # def compute_loss(logits, labels):
        #     labels = tf.reshape(labels, [-1])
        #     labels_onehot = tf.one_hot(labels, depth=n_classes)
        #     logits = tf.reshape(logits, [-1, n_classes])

        #     per_example_loss = loss_object(y_true=labels_onehot, y_pred=logits)

        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        val_loss = tf.keras.metrics.Mean(name='val_loss')
        val_acc = tf.keras.metrics.SparseCategoricalAccuracy(
            name='val_accuracy')

        # ==== Create model and optimizer ====
        model = UNet()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            init_lr, decay_steps=decay_step, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_path, max_to_keep=20, checkpoint_name='rfn-unet-v4-rice')

        # ==== Train step ====
        def train_step(inputs, labels):
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                # loss = compute_weighted_loss(logits=logits, labels=labels)
                loss = compute_weighted_loss(logits=logits, labels=labels)

                # ==== 2020-08-27 Parameter regularization  ====
                loss_regularization = []
                for p in model.trainable_variables:
                    loss_regularization.append(tf.nn.l2_loss(p))
                loss_regularization = tf.reduce_sum(
                    tf.stack(loss_regularization))

                loss = loss + 0.0001 * loss_regularization
                # ==== End  ====

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return loss

        # ==== Val step ====
        def val_step(inputs, labels):
            logits = model(inputs, training=False)
            labels = tf.reshape(labels, [-1])
            labels_onehot = tf.one_hot(labels, depth=n_classes)
            logits = tf.reshape(logits, [-1, n_classes])
            v_loss = loss_object(y_true=labels_onehot, y_pred=logits)

            val_loss.update_state(v_loss)
            val_acc.update_state(y_true=labels, y_pred=logits)

        # ==== Restore checkpoint ====
        if restore:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print('[{}] Restore checkpoint from {}'.format(
                time.asctime(), checkpoint_manager.latest_checkpoint))

        # ==== 2020-09-17 Distributed training step ====
        @tf.function
        def distributed_train_step(dataset_inputs, dataset_labels):
            per_replica_losses = strategy.run(
                train_step, args=(dataset_inputs, dataset_labels))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        # ==== 2020-09-17 Distributed val step ====
        @tf.function
        def distributed_val_step(dataset_inputs, dataset_labels):
            return strategy.run(val_step, args=(dataset_inputs, dataset_labels))

        # ==== Train loop ====
        with file_writer.as_default():
            step = 0
            for epoch in range(epochs):
                # For one epoch
                start = time.time()
                total_loss = 0.0
                num_batches = 0

                train_set = train_set.shuffle(buffer_size=buffer_size)
                train_dist_set = strategy.experimental_distribute_dataset(
                    train_set)
                for record in train_dist_set:
                    images, labels = record['x'], record['y']
                    train_loss = distributed_train_step(images, labels)
                    total_loss += train_loss
                    step += 1
                    num_batches += 1

                    tf.summary.scalar('Train loss', train_loss, step=step)
                    print('Epoch {}, step {}, train loss {:0.4f}'.format(
                        epoch + 1, step, train_loss.numpy()))

                    if step % decay_step == 0:
                        # Validation
                        val_set = val_set.shuffle(buffer_size=buffer_size)
                        val_dist_set = strategy.experimental_distribute_dataset(
                            val_set)
                        for val_record in val_dist_set:
                            images, labels = val_record['x'], val_record['y']
                            distributed_val_step(images, labels)

                        tf.summary.scalar(
                            'Val loss', val_loss.result(), step=step)
                        tf.summary.scalar(
                            'Val accuracy', val_acc.result() * 100, step=step)
                        print('[{}] Val loss {:0.4f}, val accuracy {:0.4f}'.format(
                            time.asctime(), val_loss.result(), val_acc.result() * 100))

                        val_loss.reset_states()
                        val_acc.reset_states()

                        for val_record in val_set.take(1):
                            images, labels = val_record['x'], val_record['y']

                            probs = model(images, training=False)
                            preds = tf.argmax(probs, axis=-1)

                            image, label, pred = images.numpy()[0], labels.numpy()[
                                0], preds.numpy()[0]

                            image = (image - image.min()) / (image.max() - image.min() + 1e-5)
                            label = label.astype(np.float32) / 3
                            pred = pred.astype(np.float32) / 3

                            plt.imsave(
                                os.path.join(
                                    'Visualization/', 'Epoch_{}_step_{}.png'.format(epoch + 1, step)),
                                np.hstack([
                                    image,
                                    np.broadcast_to(
                                        label[..., np.newaxis], image.shape),
                                    np.broadcast_to(
                                        pred[..., np.newaxis], image.shape)
                                ])
                            )
                            plt.close()

                # Checkpoint
                checkpoint_save_path = checkpoint_manager.save()
                print('[{}] Checkpoint saved, for step {}, at {}'.format(
                    time.asctime(), step, checkpoint_save_path))

                end = time.time()


if __name__ == "__main__":
    train_loop()
