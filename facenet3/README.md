This is the successor model, uncompress the dataset [facenet3_dataset.7z](facenet3_dataset.7z), run [`go.sh`](go.sh).

Edit the training parameters in [fit.py](https://github.com/mrbid/FaceTo3D/blob/main/facenet3/fit.py#L16).

---

A trained model from this network is [facenet3_model_1.7z](facenet3_model_1.7z). This can be used to generate 3D outputs from 2D inputs, the problem being is that it generates the same average head invariant of the input.
