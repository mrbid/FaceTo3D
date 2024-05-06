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

You can terminate a fit (training) or pred (prediction) process at any point and it will restart where it last left off. To start anew use the `cleanall` or `cleanpred` alises.
