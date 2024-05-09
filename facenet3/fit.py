# James William Fletcher (github.com/mrbid) May 2024
# GlorotUniform, ADAM, RELU/GELU, MeanSquaredError
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
layer_units = 32
batches = 6
epoches = 166
use_bias = True
dataset_size = 3333
dataset_limit = 90


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

# print(train_x.shape)
# print(train_x)
# print(train_y.shape)
# print(train_y)
# exit()


##########################################
#   TRAIN
##########################################
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
print("\n--Training Model")

# weight init
weight_init = keras.initializers.GlorotUniform() # GlorotUniform(), LecunNormal()

# construct neural network
model = Sequential()
model.add(Input(shape=(inputsize,)))
model.add(Dense(layer_units, activation=activator, use_bias=use_bias, kernel_initializer=weight_init))
if layers > 0:
    for x in range(layers):
        model.add(Dense(layer_units, use_bias=use_bias, activation=activator, kernel_initializer=weight_init))
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

# save keras model
model.save("model.keras")

# print timing
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")

# # save weights for C array
# print("")
# print("Exporting weights...")
# project = "facenet3"
# st = time_ns()
# li = 0
# f = open("model_layers.h", "w")
# f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n// accuracy: " + "{:.8f}".format(history.history['accuracy'][-1]) + "\n// loss: " + "{:.8f}".format(history.history['loss'][-1]) + "\n\n")
# if f:
#     for layer in model.layers:
#         total_layer_weights = layer.get_weights()[0].transpose().flatten().shape[0]
#         total_layer_units = layer.units
#         layer_weights_per_unit = total_layer_weights / total_layer_units
#         print("+ Layer:", li)
#         print("Total layer weights:", total_layer_weights)
#         print("Total layer units:", total_layer_units)
#         print("Weights per unit:", int(layer_weights_per_unit))

#         f.write("const float " + project + "_layer" + str(li) + "[] = {")
#         isfirst = 0
#         wc = 0
#         bc = 0
#         if layer.get_weights() != []:
#             for weight in layer.get_weights()[0].transpose().flatten():
#                 wc += 1
#                 if isfirst == 0:
#                     f.write(str(weight))
#                     isfirst = 1
#                 else:
#                     f.write("," + str(weight))
#                 if wc == layer_weights_per_unit:
#                     f.write(", /* bias */ " + str(layer.get_weights()[1].transpose().flatten()[bc]))
#                     wc = 0
#                     bc += 1
#         f.write("};\n\n")
#         li += 1
# f.write("#endif\n")
# f.close()

# # print timing
# timetaken = (time_ns()-st)/1e+9
# print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")
