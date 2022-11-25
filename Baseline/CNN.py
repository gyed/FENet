#In[]: import tensorflow2.0
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.keras.backend.clear_session()
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import datetime
import numpy as np
import pandas as pd
import random

tf.random.set_seed(1)
np.random.seed(1)
random.seed(5)

# 1 PAD*
# dataset_dir = 'dataset_3.csv'

# 2 PAD-UCDSAD
dataset_dir = 'dataset_abcx.csv'
dataset_dir2 = 'dataset_dublin.csv'

# 3 COLD-START
dataset_dir = 'dataset_D_cold_train.csv'
dataset_dir2 = 'dataset_dublin.csv'
dataset_dir3 = 'dataset_D_cold_test.csv'

train_log_dir = 'logs/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
check_callback = keras.callbacks.ModelCheckpoint(filepath='models/model_{epoch}.h5', # .format(model_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                                                 save_best_only=False, monitor='val_loss', verbose=1)
batch_size = 64



# %% Model
Input = keras.Input(shape=(60, 1), name='input')

model = keras.Sequential([
    layers.Conv1D(1, 3, padding='same', activation=tf.nn.relu),
    layers.BatchNormalization(),
    layers.MaxPool1D(2, 1, 'same'),
    layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu),
    layers.BatchNormalization(),
    layers.MaxPool1D(2, 1, 'same'),
    layers.Conv1D(64, 3, padding='same', activation=tf.nn.relu),
    layers.Flatten(),
    layers.Dense(100, activation=tf.nn.relu),
    layers.Dropout(0.5),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dropout(0.5),
    layers.Dense(2, activation='sigmoid')
])



# %% optimizer, loss, metrics
model.compile(
    optimizer='adam',
    loss=keras.losses.CategoricalCrossentropy(),
    metrics= [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
)

# %%
# 1 PAD*
# data = pd.read_csv(dataset_dir).values
# data = np.expand_dims(data, axis=2)
# nrow, _, _ = data.shape
# random.shuffle(data)
# test_head = 0
# test_tail = int(nrow/10)
# training_head = test_tail
# training_tail = int(nrow)
# training_x = data[training_head:training_tail, :-3].astype(np.float32)
# test_x = data[test_head:test_tail, :-3].astype(np.float32)
# labels = [[[int(not y), int(y)] for y in x] for x in data[:, -3:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
# labels = np.array(labels).astype(np.float32)
# training_y_0, training_y_1, training_y_2 = labels[training_head:training_tail, 0], labels[training_head:training_tail, 1], labels[training_head:training_tail, 2]
# test_y_0, test_y_1, test_y_2 = labels[test_head:test_tail, 0], labels[test_head:test_tail, 1], labels[test_head:test_tail, 2]

# 2 PAD-UCDSAD
# data1 = pd.read_csv(dataset_dir)
# print(data1.shape)
# data2 = pd.read_csv(dataset_dir2)
# print(data2.shape)
# data = pd.concat([data1,data2]).values
# print(data.shape)
# data = np.expand_dims(data, axis=2)
# nrow, _, _ = data.shape
# random.shuffle(data)
# test_head = 0
# test_tail = int(nrow/10)
# training_head = test_tail
# training_tail = int(nrow)
# training_x = data[training_head:training_tail, :-3].astype(np.float32)
# test_x = data[test_head:test_tail, :-3].astype(np.float32)
# labels = [[[int(not y), int(y)] for y in x] for x in data[:, -3:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
# labels = np.array(labels).astype(np.float32)
# training_y_0, training_y_1, training_y_2 = labels[training_head:training_tail, 0], labels[training_head:training_tail, 1], labels[training_head:training_tail, 2]
# test_y_0, test_y_1, test_y_2 = labels[test_head:test_tail, 0], labels[test_head:test_tail, 1], labels[test_head:test_tail, 2]

# 3 COLD-START
data1 = pd.read_csv(dataset_dir)
print(data1.shape)
data2 = pd.read_csv(dataset_dir2)
print(data2.shape)
data1 = np.expand_dims(data1, axis=2)
data2 = np.expand_dims(data2, axis=2)
training_x = data1[:, :-3].astype(np.float32)
test_x = data2[:, :-3].astype(np.float32)
labels = [[[int(not y), int(y)] for y in x] for x in data1[:, -3:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
labels = np.array(labels).astype(np.float32)
training_y_0, training_y_1, training_y_2 = labels[:, 0], labels[:, 1], labels[:, 2]
labels = [[[int(not y), int(y)] for y in x] for x in data2[:, -3:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
labels = np.array(labels).astype(np.float32)
test_y_0, test_y_1, test_y_2 = labels[:, 0], labels[:, 1], labels[:, 2]


data1 = pd.read_csv(dataset_dir)
print(data1.shape)
data2 = pd.read_csv(dataset_dir2)
print(data2.shape)
data = pd.concat([data1,data2]).values
print(data.shape)
data = np.expand_dims(data, axis=2)
random.shuffle(data)
data3 = pd.read_csv(dataset_dir3)
print(data3.shape)
data3 = np.expand_dims(data3, axis=2)
training_x = data[:, :-3].astype(np.float32)
test_x = data3[:, :-3].astype(np.float32)
labels = [[[int(not y), int(y)] for y in x] for x in data[:, -3:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
labels = np.array(labels).astype(np.float32)
training_y_0, training_y_1, training_y_2 = labels[:, 0], labels[:, 1], labels[:, 2]
labels = [[[int(not y), int(y)] for y in x] for x in data3[:, -3:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
labels = np.array(labels).astype(np.float32)
test_y_0, test_y_1, test_y_2 = labels[:, 0], labels[:, 1], labels[:, 2]

# %%
model.fit(training_x,
          training_y_1,
          callbacks=[
              tensorboard_callback,
              #check_callback
          ],
          validation_data = (test_x, test_y_1),
          batch_size=batch_size,
          epochs=999)


#%%
# Evaluate the model on the test data using `evaluate`
# print('\n# Evaluate on test data')
# results = model.evaluate(test_x, {'nn_block': test_y_0, 'nn_block_1': test_y_1, 'nn_block_2': test_y_2}, batch_size=batch_size)
# print('loss, l0, l1, l2, acc0, pre0, rec0, acc1, pre1, rec1, acc2, pre2, rec2', results)