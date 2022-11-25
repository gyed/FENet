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
dataset_dir = 'dataset_abcx_9.csv'
train_log_dir = 'logs_re/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=train_log_dir, histogram_freq=1)
check_callback = keras.callbacks.ModelCheckpoint(filepath='models/model_{epoch}.h5', 
                                                 save_best_only=False, monitor='val_loss', verbose=1)
batch_size = 512
FrequencySelectorDimensionality = 1

# %% Blocks
class FilterBlock(layers.Layer):
    def __init__(self):
        super(FilterBlock, self).__init__()
        self.layer1 = layers.Conv1D(1, 3, dilation_rate=2, padding='causal')
        self.layer2 = layers.Conv1D(1, 3, dilation_rate=4, padding='causal')
        self.layer3 = layers.Conv1D(1, 3, dilation_rate=8, padding='causal')
        # self.layer3 = layers.Conv1D(1, 1, padding='same')
        
    def call(self, x):
        x1 = layers.BatchNormalization()(x)
        x1 = layers.ReLU()(x1)
        x1 = self.layer1(x1)

        x2 = layers.BatchNormalization()(x1)
        x2 = layers.ReLU()(x2)
        x2 = self.layer2(x2)

        x3 = layers.BatchNormalization()(x2)
        x3 = layers.ReLU()(x3)
        x3 = self.layer3(x3)

        return x3

class FrequencySelector(layers.Layer):
    def __init__(self):
        super(FrequencySelector, self).__init__()
        self.layer1 = layers.Conv1D(FrequencySelectorDimensionality, 9, padding='same', activation=tf.nn.relu)
      
    def call(self, x):
        # x = tf.argmax(x, axis=2, name='selector')
        x = self.layer1(x)
        x = tf.reshape(x, (-1, 60, FrequencySelectorDimensionality))
        x = tf.cast(x, tf.float32)
        return x

class CnnBlock(layers.Layer):
    def __init__(self):
        super(CnnBlock, self).__init__()
        self.layer1 = layers.Conv1D(3, 3, padding='same', activation=tf.nn.relu)
        self.layer2 = layers.Conv1D(6, 3, padding='same', activation=tf.nn.relu)
        self.layer3 = layers.Conv1D(9, 3, padding='same', activation=tf.nn.relu)
        
    def call(self, x):
        x = layers.BatchNormalization()(x)
        x = self.layer1(x)
        x = layers.BatchNormalization()(x)
        x = self.layer2(x)
        x = layers.BatchNormalization()(x)
        x = self.layer3(x)
        return x

class nnBlock(layers.Layer):
    def __init__(self):
        super(nnBlock, self).__init__()
        self.layer1 = layers.Dense(60, activation=tf.nn.relu)
        self.drop1 = tf.keras.layers.Dropout(0.5)
        self.layer2 = layers.Dense(20, activation=tf.nn.relu)
        self.drop2 = tf.keras.layers.Dropout(0.5)
        self.layer3 = layers.Dense(10, activation=tf.nn.relu)
#         self.drop3 = tf.keras.layers.Dropout(0.4)
        self.layer4 = layers.Dense(2, activation='softmax')
    def call(self, x):
        flat = tf.reshape(x, [-1, 60*9])
        x = self.layer1(flat)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.drop2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

Filter = FilterBlock()
Selector = FrequencySelector()
Cnn = CnnBlock()

nn_0 = nnBlock()
nn_1 = nnBlock()
nn_2 = nnBlock()
nn_3 = nnBlock()
nn_4 = nnBlock()
nn_5 = nnBlock()
nn_6 = nnBlock()

# %% Model
Input = keras.Input(shape=(60, 1), name='input')

#x0 = layers.Conv1D(1, 3, dilation_rate=2, padding='causal', name='T-2', activation=tf.nn.relu)(Input)
x1 = layers.Conv1D(1, 3, dilation_rate=3, padding='causal', name='T-3', activation=tf.nn.relu)(Input)
x2 = layers.Conv1D(1, 3, dilation_rate=4, padding='causal', name='T-4', activation=tf.nn.relu)(Input)
x3 = layers.Conv1D(1, 3, dilation_rate=5, padding='causal', name='T-5', activation=tf.nn.relu)(Input)
x4 = layers.Conv1D(1, 3, dilation_rate=6, padding='causal', name='T-6', activation=tf.nn.relu)(Input)
# x5 = layers.Conv1D(1, 3, dilation_rate=7, padding='causal', name='T-7', activation=tf.nn.relu)(Input)

#x0 = Filter(x0)
x1 = Filter(x1)
x2 = Filter(x2)
x3 = Filter(x3)
x4 = Filter(x4)
# x5 = Filter(x5)

#x0 = layers.add([x0, Input])
x1 = layers.add([x1, Input])
x2 = layers.add([x2, Input])
x3 = layers.add([x3, Input])
x4 = layers.add([x4, Input])
# x5 = layers.add([x5, Input])

x = layers.concatenate([x1, x2, x3, x4], name='conc')
x = Selector(x)
x = Cnn(x)

n0 = nn_0(x)
n1 = nn_1(x)
n2 = nn_2(x)
n3 = nn_3(x)
n4 = nn_4(x)
n5 = nn_5(x)
n6 = nn_6(x)

model = keras.Model(inputs=[Input], outputs=[n0, n1, n2, n3, n4, n5, n6])
# keras.utils.plot_model(model, 'FTN_model.png', show_shapes=True)
# %% optimizer, loss, metrics
# ['nn_block', 'nn_block_1', 'nn_block_2']
model.compile(
    optimizer='adam',
    loss={'nn_block': keras.losses.CategoricalCrossentropy(),
          'nn_block_1': keras.losses.CategoricalCrossentropy(),
          'nn_block_2': keras.losses.CategoricalCrossentropy(),
          'nn_block_3': keras.losses.CategoricalCrossentropy(),
          'nn_block_4': keras.losses.CategoricalCrossentropy(),
          'nn_block_5': keras.losses.CategoricalCrossentropy(),
          'nn_block_6': keras.losses.CategoricalCrossentropy()},
    metrics= {'nn_block': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
              'nn_block_1': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
              'nn_block_2': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
              'nn_block_3': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
              'nn_block_4': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
              'nn_block_5': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)],
              'nn_block_6': [keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(class_id=1), keras.metrics.Recall(class_id=1)]},
    loss_weights=[1, 1, 1, 1, 1, 1, 1]
)
# %%
data = pd.read_csv(dataset_dir).iloc[:, :69].values
data = np.expand_dims(data, axis=2)
nrow, _, _ = data.shape
random.shuffle(data)

test_head = 0
test_tail = int(nrow/10)
training_head = test_tail
training_tail = int(nrow)

training_x = data[training_head:training_tail, :-9].astype(np.float32)
test_x = data[test_head:test_tail, :-9].astype(np.float32)

labels = [[[int(not y), int(y)] for y in x] for x in data[:, -9:]] # [0, 1, 0] -> [[1, 0], [0, 1], [1, 0]]
labels = np.array(labels).astype(np.float32)

def TrainL(offset):
    return labels[training_head:training_tail, offset+4]

def TestL(offset):
    return labels[test_head:test_tail, offset+4]  

model.fit(training_x,
          {'nn_block': TrainL(-3), 'nn_block_1': TrainL(-2), 'nn_block_2': TrainL(-1), 'nn_block_3': TrainL(0), 'nn_block_4': TrainL(1), 'nn_block_5': TrainL(2), 'nn_block_6': TrainL(3)},
          callbacks=[
              tensorboard_callback,
              #check_callback
          ],
          validation_data = (test_x, {'nn_block': TestL(-3), 'nn_block_1': TestL(-2), 'nn_block_2': TestL(-1), 'nn_block_3': TestL(0), 'nn_block_4': TestL(1), 'nn_block_5': TestL(2), 'nn_block_6': TestL(3)}),
          batch_size=batch_size,
          epochs=999999)
#%%
# Evaluate the model on the test data using `evaluate`
# print('\n# Evaluate on test data')
# results = model.evaluate(test_x, {'nn_block': test_y_0, 'nn_block_1': test_y_1, 'nn_block_2': test_y_2}, batch_size=batch_size)
# print('loss, l0, l1, l2, acc0, pre0, rec0, acc1, pre1, rec1, acc2, pre2, rec2', results)