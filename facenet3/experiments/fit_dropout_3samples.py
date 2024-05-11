# James William Fletcher (github.com/mrbid) May 2024
# GlorotUniform/LecunNormal, Adam/Adamax, ReLU/GeLU, MeanSquaredError
# ---
# In this script dropout of 0.8 is used on just three training samples
# this is a very large dropout value and yet without this dropout the
# network fails to learn a good grayscale representation of the head.
# This seems to scale badly with more samples added to the training
# process but it is a very interesting effect.
# ---
import sys
import os
import numpy as np
from time import time_ns
from sys import exit
from os.path import isfile

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # cpu training only

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
inputsize = 1024
outputsize = 32768
learning_rate = 0.001 # 0.001 is the most stable
activator = 'gelu' # selu, gelu, mish, relu, relu6, silu, hard_silu, hard_sigmoid, softsign, softplus, softmax, sigmoid, leaky_relu, tanh, elu
layers = 6
layer_units = 64
batches = 3
epoches = 777
dropout = 0.8
use_bias = True
dataset_size = 3333
dataset_limit = 3


##########################################
#   LOAD DATASET
##########################################
print("\n--Loading Dataset")
st = time_ns()

print("Dataset Elements:", "{:,}".format(dataset_size))

if isfile("train_x.npy"):
    train_x = np.load("train_x.npy")
    train_y = np.load("train_y.npy")
else:
    load_x = []
    with open("train_x.dat", 'rb') as f:
        load_x = np.fromfile(f, dtype=np.uint8)
    load_x = load_x / 255
    train_x = np.reshape(load_x, [dataset_size, inputsize])

    load_y = []
    with open("train_y.dat", 'rb') as f:
        load_y = np.fromfile(f, dtype=np.uint8)
    load_y = load_y / 255
    train_y = np.reshape(load_y, [dataset_size, outputsize])
    
    np.save("train_x.npy", train_x)
    np.save("train_y.npy", train_y)

if dataset_limit > 0:
    print("Dataset Limit/Load Size:", "{:,}".format(dataset_limit))
    train_x = train_x[:dataset_limit]
    train_y = train_y[:dataset_limit]

timetaken = (time_ns()-st)/1e+9
print("Time Taken:", "{:.2f}".format(timetaken), "seconds")


##########################################
#   TRAIN
##########################################
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
print("\n--Training Model")

# weight init
weight_init = keras.initializers.GlorotUniform() # GlorotUniform(), LecunNormal()

# construct neural network
model = Sequential()
model.add(Input(shape=(inputsize,)))
model.add(Dense(layer_units, activation=activator, use_bias=use_bias, kernel_initializer=weight_init))
model.add(Dropout(dropout))
if layers > 0:
    for x in range(layers):
        model.add(Dense(layer_units, use_bias=use_bias, activation=activator, kernel_initializer=weight_init))
        model.add(Dropout(dropout))
model.add(Dense(outputsize, use_bias=use_bias, kernel_initializer=weight_init))

# output summary
model.summary()

# compile network
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['accuracy'])

# train network
history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches, shuffle=True)
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")


##########################################
#   EXPORT
##########################################

# export model
print("\n--Exporting Model")
st = time_ns()
model.save("model.keras")
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")



