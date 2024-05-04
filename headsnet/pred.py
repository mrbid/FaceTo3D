# github.com/mrbid
import sys
import os
import random
import struct
import numpy as np
from os import mkdir
from os.path import isdir
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = ""
predict_points = 200000
if len(sys.argv) >= 2: model_path = sys.argv[1]
if len(sys.argv) >= 3: predict_points = int(sys.argv[2])
model = keras.models.load_model(model_path)

predict_x = np.empty([predict_points, 2], float)
rseed = random.uniform(0,1)
for i in range(predict_points): predict_x[i] = [rseed, i/predict_points]
p = model.predict(predict_x)

if not isdir('pred'): mkdir('pred')

np.asarray(p).tofile('pred/head_sample.csv', sep=',')

# export PLY file of prediction
# https://gist.github.com/Shreeyak/9a4948891541cb32b501d058db227fff
fid = open("pred/head_sample.ply", 'wb')
fid.write(bytes('ply\n', 'utf-8'))
fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
fid.write(bytes('element vertex ' + str(predict_points) + '\n', 'utf-8'))
fid.write(bytes('property float x\n', 'utf-8'))
fid.write(bytes('property float y\n', 'utf-8'))
fid.write(bytes('property float z\n', 'utf-8'))
fid.write(bytes('property uchar red\n', 'utf-8'))
fid.write(bytes('property uchar green\n', 'utf-8'))
fid.write(bytes('property uchar blue\n', 'utf-8'))
fid.write(bytes('end_header\n', 'utf-8'))
for i in range(predict_points):
    fid.write(bytearray(struct.pack("fffccc",p[i][0],p[i][1],p[i][2],
        bytes([np.uint8(p[i][3]*255)]),
        bytes([np.uint8(p[i][4]*255)]),
        bytes([np.uint8(p[i][5]*255)]),
        )))
fid.close()

# fid = open("head_sample.ply", 'wb')
# fid.write(bytes('ply\n', 'utf-8'))
# fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
# fid.write(bytes('element vertex ' + str(predict_points) + '\n', 'utf-8'))
# fid.write(bytes('property float x\n', 'utf-8'))
# fid.write(bytes('property float y\n', 'utf-8'))
# fid.write(bytes('property float z\n', 'utf-8'))
# fid.write(bytes('property float red\n', 'utf-8'))
# fid.write(bytes('property float green\n', 'utf-8'))
# fid.write(bytes('property float blue\n', 'utf-8'))
# fid.write(bytes('end_header\n', 'utf-8'))
# for i in range(predict_points):
#     fid.write(bytearray(struct.pack("ffffff",p[i][0],p[i][1],p[i][2],p[i][3],p[i][4],p[i][5])))
# fid.close()
