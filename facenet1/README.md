## generate new dataset

1. run `get_faces.sh` and wait for it to finish
2. run `python3 gen_meshes.py` and wait for it to finish

You will now have a `ply` directory full of 3D heads.

3. run `python3 jpg_to_pgm.py` and wait for it to finish
4. run `remove_pgm_header.sh` and wait for it to finish
5. run `faces3_to_trainx.sh` and wait for it to finish

You now have a `train_x.dat`.

In the directory `train_y_gen/` is the C program that will read the `ply` directory and create a `train_y.dat` of 32^3 voxel volumes from the 3D meshes of heads.

Once you have your `train_x.dat` and `train_y.dat` you are ready to train a network, `train_x` is the inputs and `train_y` is the target outputs. These files are in uint8 format.

## use pre-generated dataset
Unzip [facenet_dataset.7z](facenet_dataset.7z).

## index

- `fit.py` - train network
- `pred.py <input_image_path> <model_path>` - predict using a trained network model
