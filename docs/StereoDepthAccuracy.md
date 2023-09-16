

# Depth estimation from disparity 

"Stereo Vision - Simplified Case". Classic case with identical aligned cameras.

Z: depth
B: baseline
f: focal length
d: disparity (horizontal distance between corresponding pixels on left and right images)

Then you get
```
Z = b*f/d
```

Assume you have a variance `variance_d` for the disparity, where `variance_d = sigma_d^2`.
We want to compute `variance_Z`, where  `variance_Z = sigma_Z^2`.

From the above formula `Z = b*f/d`, if one differentiate it we get 
```
deltaZ = -b*f/(d^2) * deltad
```
Then, considering deltas around the mean terms, one has
```
variance_Z = E[delta_Z * delta_Z^T] = (b*f)^2/(d^4) * variance_d
```
and therefore (by root-squaring)
```
sigma_Z = b*f/(d^2) * sigma_d =  (Z^2/(B*f)) * sigma_d
```
This entails that the sigma of Z increase quadratically with the depth (assuming an almost constant sigma_d).
