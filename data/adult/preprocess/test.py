import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import os
import pandas as pd
import json

parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
def main():
    r = 0


# def gan_indices(df,indices,size = 12):
#     gan_ind = []
#     cnt1, cnt2, cnt3, cnt4 = 0,0,0,0
#     i = 0
#     while len(gan_ind) < size:
#         if df['gender'].loc[indices[i]]==0 and df['race'].loc[indices[i]]==0 and cnt1 < 3:
#             gan_ind.append(indices[i])
#             cnt1 += 1
#         elif df['gender'].loc[indices[i]]==0 and df['race'].loc[indices[i]]==1 and cnt2 < 3:
#             gan_ind.append(indices[i])
#             cnt2 += 1
#         elif df['gender'].loc[indices[i]]==1 and df['race'].loc[indices[i]]==0 and cnt3 < 3:
#             gan_ind.append(indices[i])
#             cnt3 += 1
#         elif df['gender'].loc[indices[i]]==1 and df['race'].loc[indices[i]]==1 and cnt4 < 3:
#             gan_ind.append(indices[i])
#             cnt4 += 1
#         i += 1
#     print(df.loc[gan_ind])
#     print(gan_ind)
#     return gan_ind

# def test1():
    
#     teston = get_indices_workers_het(49000,1500)
#     print("Shapes : ", len(teston))
#     tot = 0
#     his = []
#     for i in range(len(teston)):
#         a= teston[i].shape[0]
#         his.append(a)
#         tot += a
#     print("Total = ", tot)
#     plt.hist(his,bins = 5, color = 'yellow',
#             edgecolor = 'red')
#     plt.xlabel('valeurs')
#     plt.ylabel('nombres')
#     plt.title('Exemple d\' histogramme simple')
#     plt.show()


# def get_indices_workers_het(ns, num_workers): #Generate the samples indices for each worker
#     num_samples = ns
#     mean1 = (num_samples / (2*0.8*num_workers)) #80% of users will hold 50% of data
#     mean2 = (num_samples / (2*0.2*num_workers)) #20% of the rest will hold the 50% remaining data

#     print("Mean 1 : ", mean1)
#     print("Mean 2 : ", mean2)
#     indices = np.arange(ns) # Indices of the samples
#     np.random.shuffle(indices)

#     data_remaining = num_samples # Number of samples undistributed remaining in the dataset
#     num_samples_workers =  [0]
#     for i in range(num_workers-1):
#         if i < 0.8*num_workers + 1: # 80% users' data
#             num_samples_i = np.random.poisson(mean1)
#             if data_remaining - num_samples_i < 0:
#                 num_samples_i = data_remaining
#                 break
#             num_samples_workers.append(num_samples_i)
#             data_remaining -= num_samples_i
#         else: # 20% remaining users' data
#             num_samples_i = np.random.poisson(mean2)
#             if data_remaining - num_samples_i < 0:
#                 num_samples_i = data_remaining
#                 break

#             num_samples_workers.append(num_samples_i)
#             data_remaining -= num_samples_i

#     num_samples_workers.append(data_remaining)
#     make_indices = np.cumsum(num_samples_workers)
#     indices_workers = [indices[make_indices[i]:make_indices[i+1]] for i in range(len(make_indices)-1)] #For each worker, we have a list of the samples he possesses
#     return indices_workers


# def ttttt():
    

#     features = np.array(data['x'])
#     labels = data['y']

#     # privileged = 0 * features[:,0] + 1 #Privileged attributes is denoted by '1'
#     # unprivileged = 0 * features[:,0]

#     # unpriv_samples = tf.equal(features[:,0],unprivileged)
#     # priv_samples = tf.equal(features[:,0],privileged)

#     logits = tf.layers.dense(features, 12, activation=tf.nn.relu)
#     logits1 = tf.layers.dense(logits, 2, activation=tf.nn.sigmoid)


#     loss1 = tf.compat.v1.keras.losses.CategoricalCrossentropy(from_logits=True)(tf.one_hot(labels,2), logits1, sample_weight = [100,1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

#     for _ in range(100):
#         tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0003).minimize(
#             loss=loss1)

#     predictions1 = tf.argmax(logits1, axis=-1)
#     correct_pred1 = tf.equal(predictions1, labels)

#     # unpriv_pred1 = predictions1[unpriv_samples]
#     # temp = tf.count_nonzero(unpriv_pred1 * 0 + 1)
#     # term11 = tf.count_nonzero(unpriv_pred1) / temp

#     # priv_pred1 = predictions1[priv_samples]
#     # temp = tf.count_nonzero(priv_pred1 * 0 + 1)
#     # term21 = tf.count_nonzero(priv_pred1) / temp


#     # DI1 = tf.cond(tf.equal(tf.math.divide_no_nan(term11,term21),tf.constant(0.0, dtype = tf.float64)), 
#     #     true_fn = lambda : tf.constant(1.0, dtype = tf.float64), 
#     #     false_fn = lambda : tf.math.divide_no_nan(term11,term21))


#     init = tf.compat.v1.global_variables_initializer()
    
#     with tf.compat.v1.Session() as sess:  
#         sess.run(init)
#         d = sess.run(logits1)
#         c = sess.run(correct_pred1)
#         print(c)
#         print(d)

# def get_sample_weights(data, sensitive_attribute):
#     data_x = np.array(data['x'])
#     data_y = np.array(data['y'])

#     num_samples = data_y.shape[0]
#     sample_weights = np.zeros(num_samples)

#     # Count positive/negative and privileged/unprivileged samples
#     pos_features = data_x[data_y == 1]
#     neg_features = data_x[data_y == 0]

#     pos_labels = data_y[data_y == 1]
#     neg_labels = data_y[data_y == 0]

#     priv_features = data_y[data_x[:,sensitive_attribute] == 1]
#     unpriv_features = data_y[data_x[:,sensitive_attribute] == 0]

#     # Split samples by labels and sensitive attributes
    
#     pos_unpriv_indices = np.equal(pos_features[:,sensitive_attribute],np.zeros(pos_features.shape[0]))
#     pos_priv_indices = np.equal(pos_features[:,sensitive_attribute],np.ones(pos_features.shape[0]))
#     neg_unpriv_indices = np.equal(neg_features[:,sensitive_attribute],np.zeros(neg_features.shape[0]))
#     neg_priv_indices = np.equal(neg_features[:,sensitive_attribute],np.ones(neg_features.shape[0]))

#     pos_unpriv_labels = pos_labels[pos_unpriv_indices]
#     pos_priv_labels = pos_labels[pos_priv_indices]
#     neg_unpriv_labels = neg_labels[neg_unpriv_indices]
#     neg_priv_labels = neg_labels[neg_priv_indices]

#     cnt_pos, cnt_neg = len(pos_features), len(neg_features)
#     cnt_unpriv, cnt_priv = len(unpriv_features), len(priv_features)
#     cnt_pos_unpriv, cnt_pos_priv = len(pos_unpriv_labels), len(pos_priv_labels)
#     cnt_neg_unpriv, cnt_neg_priv = len(neg_unpriv_labels), len(neg_priv_labels)

#     for ind in range(num_samples):
#         if data_x[ind,sensitive_attribute] == 0 and data_y[ind] == 0:
#             sample_weights[ind] = cnt_neg * cnt_unpriv / (cnt_neg_unpriv * num_samples)
#         elif data_x[ind,sensitive_attribute] == 0 and data_y[ind] == 1:
#             sample_weights[ind] = cnt_pos * cnt_unpriv / (cnt_pos_unpriv * num_samples)
#         elif data_x[ind,sensitive_attribute] == 1 and data_y[ind] == 0:
#             sample_weights[ind] = cnt_neg * cnt_priv / (cnt_neg_priv * num_samples)
#         elif data_x[ind,sensitive_attribute] == 1 and data_y[ind] == 1:
#             sample_weights[ind] = cnt_pos * cnt_priv / (cnt_pos_priv * num_samples)
        
#     return sample_weights


if __name__ == '__main__':
	main()