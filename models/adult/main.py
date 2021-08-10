import numpy as np
import os
import sys
import tensorflow as tf

def main():
    create_model()

def create_model():
    features = tf.compat.v1.placeholder(tf.float32, [None, 10])
    labels = tf.compat.v1.placeholder(tf.int64, [None])

    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape = (None, 10)))
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))


    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate = 0.03), 
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']) #metrics TP TN FP FN dispo

    sample_weight = np.ones(shape=(labels.shape[0],))
    sample_weight[labels == 0] = 1
    sample_weight[labels == 1] = 1
    
    model.fit(features, labels, batch_size = labels.shape[0], sample_weights = sample_weight, epochs = 1)

if __name__ == '__main__':
	main()