# github.com/mrbid
import sys
import os
import numpy as np
from PIL import Image
from time import time_ns
from sys import exit
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = ""
if os.path.isfile("pred_input.npy"):
    predict_x = np.load("pred_input.npy")
    if len(sys.argv) >= 2: model_path = sys.argv[1]
else:
    image_path = ""
    if len(sys.argv) >= 2: image_path = sys.argv[1]
    image = Image.open(image_path)
    image = image.resize((32, 32), Image.Resampling.LANCZOS)
    image = image.convert('L')
    image.save("/tmp/input.pgm", "PPM")
    image.close()
    with open('/tmp/input.pgm', 'rb') as in_file:
        with open('pred_input.dat', 'wb') as out_file:
            out_file.write(in_file.read()[13:])
    os.remove("/tmp/input.pgm")

    load_x = []
    with open("pred_input.dat", 'rb') as f:
        load_x = np.fromfile(f, dtype=np.uint8)
    load_x = load_x / 255
    predict_x = np.reshape(load_x, [-1, 1024])
    np.save("pred_input.npy", predict_x)
    os.remove("pred_input.dat")
    print("input generated, next run specify model as argv.")
    exit()

from tensorflow import keras

mps = Path(model_path).stem
os.makedirs("pred_multi", exist_ok=True)
model = keras.models.load_model(model_path)
p = model.predict(predict_x)
npa = np.asarray(p)
npa[npa < 0.003] = 0
npa = npa * 255
npa = npa.round().astype(int)
with open("pred_multi/" + mps + ".csv", "w") as f:
    f.write(str(npa[0][0]) + ",")
with open("pred_multi/" + mps + ".dat", "wb") as f:
    f.write(np.uint8(npa[0]).tobytes())

