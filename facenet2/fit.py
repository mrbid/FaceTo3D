# github.com/mrbid
import sys
import os
import numpy as np
from time import time_ns
from sys import exit
from os import mkdir
from os.path import isdir
from os.path import isfile
from pathlib import Path

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
activator = 'selu'
layers = 4
layer_units = 32
batches = 9
epoches = 333
use_bias = True
dataset_size = 3333

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
from tensorflow.keras.layers import Dense
print("\n--Training Model")

# construct neural network
model = Sequential()
model.add(Dense(layer_units, activation=activator, use_bias=use_bias, input_dim=inputsize))
if layers > 0:
    for x in range(layers):
        model.add(Dense(layer_units, use_bias=use_bias, activation=activator))
model.add(Dense(outputsize, use_bias=use_bias))

# output summary
model.summary()

if optimiser == 'adam':
    optim = keras.optimizers.Adam(learning_rate=0.001)
elif optimiser == 'sgd':
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.3, decay_steps=epoches*dataset_size, decay_rate=0.1)
    #lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.1, decay_steps=epoches*dataset_size, decay_rate=0.01)
    optim = keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False)
elif optimiser == 'momentum':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
elif optimiser == 'nesterov':
    optim = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
elif optimiser == 'nadam':
    optim = keras.optimizers.Nadam(learning_rate=0.001)
elif optimiser == 'adagrad':
    optim = keras.optimizers.Adagrad(learning_rate=0.001)
elif optimiser == 'rmsprop':
    optim = keras.optimizers.RMSprop(learning_rate=0.001)
elif optimiser == 'adadelta':
    optim = keras.optimizers.Adadelta(learning_rate=0.001)
elif optimiser == 'adamax':
    optim = keras.optimizers.Adamax(learning_rate=0.001)
elif optimiser == 'ftrl':
    optim = keras.optimizers.Ftrl(learning_rate=0.001)

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
if not isdir('models'): mkdir('models')
if not isdir(model_name): mkdir(model_name)
st = time_ns()

# save keras model
model.save(model_name + ".keras")

# print timing
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")

# save weights for C array
print("")
print("Exporting weights...")
st = time_ns()
li = 0
f = open(model_name + "/" + project + "_layers.h", "w")
f.write("#ifndef " + project + "_layers\n#define " + project + "_layers\n\n// accuracy: " + "{:.8f}".format(history.history['accuracy'][-1]) + "\n// loss: " + "{:.8f}".format(history.history['loss'][-1]) + "\n\n")
if f:
    for layer in model.layers:
        total_layer_weights = layer.get_weights()[0].transpose().flatten().shape[0]
        total_layer_units = layer.units
        layer_weights_per_unit = total_layer_weights / total_layer_units
        print("+ Layer:", li)
        print("Total layer weights:", total_layer_weights)
        print("Total layer units:", total_layer_units)
        print("Weights per unit:", int(layer_weights_per_unit))

        f.write("const float " + project + "_layer" + str(li) + "[] = {")
        isfirst = 0
        wc = 0
        bc = 0
        if layer.get_weights() != []:
            for weight in layer.get_weights()[0].transpose().flatten():
                wc += 1
                if isfirst == 0:
                    f.write(str(weight))
                    isfirst = 1
                else:
                    f.write("," + str(weight))
                if wc == layer_weights_per_unit:
                    f.write(", /* bias */ " + str(layer.get_weights()[1].transpose().flatten()[bc]))
                    wc = 0
                    bc += 1
        f.write("};\n\n")
        li += 1
f.write("#endif\n")
f.close()

# print timing
timetaken = (time_ns()-st)/1e+9
print("\nTime Taken:", "{:.2f}".format(timetaken), "seconds\n")
