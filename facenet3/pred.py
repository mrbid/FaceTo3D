# github.com/mrbid
import sys
import os
import numpy as np
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_path = "model.keras"
image_path = "input_face.jpg"
if len(sys.argv) >= 2: image_path = sys.argv[1]
if len(sys.argv) >= 3: model_path = sys.argv[2]

image = Image.open(image_path)
image = image.resize((32, 32), Image.Resampling.LANCZOS)
image = image.convert('L')
image.save("/tmp/input.pgm", "PPM")
image.close()
with open('/tmp/input.pgm', 'rb') as in_file:
    with open('/tmp/pred_input.dat', 'wb') as out_file:
        out_file.write(in_file.read()[13:])
os.remove("/tmp/input.pgm")

load_x = []
with open("/tmp/pred_input.dat", 'rb') as f:
    load_x = np.fromfile(f, dtype=np.uint8)
os.remove("/tmp/pred_input.dat")
load_x = load_x / 255
predict_x = np.reshape(load_x, [-1, 1024])

from tensorflow import keras
model = keras.models.load_model(model_path)
p = model.predict(predict_x)
npa = np.asarray(p)
npa[npa < 0.003] = 0
npa = npa * 255
npa = npa.round().astype(int)
npa.tofile('predicted.csv', sep=',')

