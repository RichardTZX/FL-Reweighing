"""Logistic regression."""

import numpy as np
import os
import sys
import tensorflow as tf
import math

from baseline_constants import ACCURACY_KEY

from model import Model
from fedprox import PerturbedGradientDescent
from utils.model_utils import batch_data, batch_data_with_weights, batch_data_binary_oversampling, batch_data_binary_oversampling_with_weights, get_sample_weights


class ClientModel(Model):

    def __init__(self, seed, lr, num_classes, input_dim, cfg=None):
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.cfg = cfg
        self.oversample = False

        self.model_name = os.path.abspath(__file__)

        if self.cfg.sensitive_attribute == 'race':
            self.sensitive_attribute = 0
        else: #cfg.sensitive_attribute == 'gender'
            self.sensitive_attribute = 1
        
        if cfg.fedprox:
            super(ClientModel, self).__init__(seed, lr, optimizer=PerturbedGradientDescent(lr, cfg.fedprox_mu))
        else:
            super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        features = tf.placeholder(tf.float32, [None, self.input_dim])
        labels = tf.placeholder(tf.int64, [None])
        self.sample_weights = tf.placeholder(tf.float32, [None])

        logits = tf.layers.dense(features, self.num_classes, kernel_regularizer = 'l2')

        loss = tf.compat.v1.keras.losses.CategoricalCrossentropy(from_logits=True)(tf.one_hot(labels,2), logits, sample_weight = self.sample_weights)

        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        predictions = tf.argmax(logits, axis=-1)

        privileged = 0 * features[:,0] + 1 #Privileged attributes is denoted by '1'
        unprivileged = 0 * features[:,0]

        unpriv_samples = tf.equal(features[:,self.sensitive_attribute],unprivileged)
        priv_samples = tf.equal(features[:,self.sensitive_attribute],privileged)

        unpriv_pred = predictions[unpriv_samples]
        temp = tf.count_nonzero(unpriv_pred * 0 + 1)
        term1 = tf.count_nonzero(unpriv_pred) / temp


        priv_pred = predictions[priv_samples]
        temp = tf.count_nonzero(priv_pred * 0 + 1)
        term2 = tf.count_nonzero(priv_pred) / temp

        correct_pred = tf.equal(predictions, labels)

        # This line is useful to prevent "NaN" or "inf" values
        DI = tf.cond(tf.equal(tf.math.divide_no_nan(term1,term2),tf.constant(0.0, dtype = tf.float64)), 
            true_fn = lambda : tf.constant(0.0, dtype = tf.float64), 
            false_fn = lambda : tf.math.divide_no_nan(term1,term2))

        eval_metric_ops = [tf.count_nonzero(correct_pred),DI,
            [tf.count_nonzero(unpriv_pred), tf.count_nonzero(unpriv_pred * 0 + 1), tf.count_nonzero(priv_pred), tf.count_nonzero(priv_pred * 0 + 1)]]
        
        return features, labels, train_op, eval_metric_ops, tf.reduce_mean(loss)

    
    def train(self, data, num_epochs=1, batch_size=10):
        """
        Trains the client model.
        Args:
            data: Dict of the form {'x': [list], 'y': [list]}.
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: Number of FLOPs computed while training given data
            update: List of np.ndarray weights, with each weight array
                corresponding to a variable in the resulting graph
        """
        '''
        train_reslt = self.test(data)
        acc = train_reslt[ACCURACY_KEY]
        loss = train_reslt['loss']
        logger.info('before: {}'.format(loss))
        '''
        data_sample_weights = get_sample_weights(data, self.sensitive_attribute)
        params_old= self.get_params()
        loss_old = self.test(data)['loss']
        
        for i in range(num_epochs):
            self.run_epoch(data, data_sample_weights, batch_size)

        train_reslt = self.test(data)
        acc = train_reslt[ACCURACY_KEY]
        loss = train_reslt['loss']
        disp_imp = train_reslt['disparate_impact']
        
        update = self.get_params()
        comp = num_epochs * math.ceil(len(data['y'])/batch_size) * batch_size * self.flops

        grad = []
        for i in range(len(update)):
            grad.append((params_old[i] - update[i]) / self.lr)
        return comp, update, acc, loss, disp_imp, grad, loss_old

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)


    def run_epoch(self, data, data_sample_weights, batch_size):
        # We use oversampling in order to not have a single label predicted
        # Label '1' is the minority label in the adult dataset.
        if self.oversample:
            batches = batch_data_binary_oversampling_with_weights(data, data_sample_weights, batch_size, seed=self.seed, label_to_oversample = 1)
        else:
            batches = batch_data_with_weights(data, data_sample_weights, batch_size, seed=self.seed)

        for batched_x, batched_y, batch_weights in batches: 

            input_data = self.process_x(batched_x)
            target_data = self.process_y(batched_y)

            self.last_features = input_data
            self.last_labels = target_data

            with self.graph.as_default():
                _, metrics, loss = self.sess.run(
                    [self.train_op, self.eval_metric_ops, self.loss],
                    feed_dict={
                        self.features: input_data,
                        self.labels: target_data,
                        self.sample_weights: batch_weights})

        acc = float(metrics[0]) / input_data.shape[0]
        return {'acc': acc, 'loss': loss, 'disparate impact': metrics[1], 'global_di': metrics[2]}

    def test(self, data):
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        data_sample_weights = get_sample_weights(data, self.sensitive_attribute)

        with self.graph.as_default():
            metrics, loss = self.sess.run(
                [self.eval_metric_ops, self.loss],
                feed_dict={
                    self.features: x_vecs, 
                    self.labels: labels,
                    self.sample_weights: data_sample_weights
                })
        acc = float(metrics[0]) / len(x_vecs)
        return {'accuracy': acc, 'loss': loss, 'disparate_impact': metrics[1], 'global_di': metrics[2]}
