# James William Fletcher (github.com/mrbid) May 2024
import sys
import os
import numpy as np
from time import time_ns
from sys import exit
from pathlib import Path
from os.path import isfile

# disable warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # cpu training only

# print everything / no truncations
np.set_printoptions(threshold=sys.maxsize)

# hyperparameters
project = "facenet"
model_name = 'keras_model'
optimiser = 'adam'
inputsize = 1024
outputsize = 1
activator = 'gelu'
layers = 6
layer_units = 32
batches = 6
epoches = 166
use_bias = True
dataset_size = 3333
dataset_limit = 90

# load option
tylfn = ""
argc = len(sys.argv)
if argc >= 2:
    tylfn = sys.argv[1]
else:
    print("!!! Please specify train_y load file path as argv parameter.")
    exit()

project = Path(tylfn).stem
model_name = 'models/' + project

##########################################
#   LOAD DATASET
##########################################
print("\n--Loading Dataset")
st = time_ns()

print("Dataset Elements:", "{:,}".format(dataset_size))
if isfile("train_x.npy"): train_x = np.load("train_x.npy")

npy_tylfn = "train_y_npy/" + project + ".npy"
if isfile(npy_tylfn):
    train_y = np.load(npy_tylfn)
else:
    load_y = []
    with open(tylfn, 'rb') as f:
        load_y = np.fromfile(f, dtype=np.uint8)
    load_y = load_y / 255
    train_y = np.reshape(load_y, [dataset_size, outputsize])
    np.save(npy_tylfn, train_y)

# tunc
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

# construct neural network
model = Sequential()
model.add(Input(shape=(inputsize,)))
model.add(Dense(layer_units, activation=activator, use_bias=use_bias))
if layers > 0:
    for x in range(layers):
        model.add(Dense(layer_units, use_bias=use_bias, activation=activator))
model.add(Dense(outputsize, use_bias=use_bias))

# output summary
model.summary()

if optimiser == 'adam':
    optim = keras.optimizers.Adam(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)

model.compile(optimizer=optim, loss='mean_squared_error', metrics=['accuracy'])

# train network
history = model.fit(train_x, train_y, epochs=epoches, batch_size=batches, shuffle=True)
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds")


##########################################
#   EXPORT
##########################################

# export model
print("\n--Exporting Model")
os.makedirs('models', exist_ok=True)
st = time_ns()

# save keras model
model.save(model_name + ".keras")

# print timing
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")

# # save weights for C array
# print("")
# print("Exporting weights...")
# os.makedirs(model_name, exist_ok=True)
# st = time_ns()
# li = 0
# f = open(model_name + "/" + project + "_layers.h", "w")
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
