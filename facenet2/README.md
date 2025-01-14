## how to use
```
source alias
fit
pred
prev
```

## start a new model & prediction
```
source alias
cleanall
```

## start a new prediction
```
source alias
cleanpred
```

Training parameters are configured in [fit.py](https://github.com/mrbid/FaceTo3D/blob/main/facenet2/fit.py#L19).

You can terminate a fit _(training)_ or pred _(prediction)_ process at any point and it will restart where it last left off. To start anew use the `cleanall` or `cleanpred` alises.

---

To use the pre-generated dataset uncompress [facenet2_dataset_npy.7z](facenet2_dataset_npy.7z).

---

A prediction generated by this network is [facenet2_prediction_1.7z](https://github.com/mrbid/FaceTo3D/raw/main/facenet2/facenet2_prediction_1.7z) the full model was 13.2 GiB.
