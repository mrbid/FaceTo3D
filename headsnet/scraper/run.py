# https://github.com/mrbid
import os
import secrets
import requests
import numpy as np
import rembg
import torch
import trimesh
from PIL import Image
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
import time

device = "cpu"
model = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(8192)
model.to(device)
rembg_session = rembg.new_session()
os.makedirs("ply", exist_ok=True)

while True:
    tt = time.time()
    image = remove_background(Image.open(requests.get("https://thispersondoesnotexist.com/", stream=True).raw), rembg_session)
    image = resize_foreground(image, 0.85)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))
    with torch.no_grad(): scene_codes = model([image], device=device)
    mesh = model.extract_mesh(scene_codes, resolution=256)[0]
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
    mesh.export("ply/" + secrets.token_urlsafe(16) + ".ply")
    print("tt: " + str(time.time()-tt))
