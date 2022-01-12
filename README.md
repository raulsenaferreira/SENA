# SHUM
Repository of the paper "All that glitters is not gold: a pragmatic metric to avoid spurious detections of out-of-distribution data"

SHUM It is a metric generated from four other metrics: "Similarity, hardness, uniqueness, and mistakennes".

The four metrics above are provided by the open source tool Fiftyone Brain (https://voxel51.com/docs/fiftyone/api/fiftyone.brain.html#) 

We adapted all 4 metrics in one single metric and we also adapted it to be applied during operation in safe-critical systems.

OOD data is usually source of errors in classifiers and object detectors built with deep learning.
Literature usually apply detectors at real-time aiming at detect and avoid such data.
However, not all OOD data exposed to the ML model will yield wrong predictions. 
To avoid spurious detections or spurious data processing at real-time, we developed this metric, which the goal is to give a clue of when an image (ID or OOD) is more error-prune to the model.
