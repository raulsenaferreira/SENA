# SHINE
Repository of the paper "All that glitters is not gold: avoiding spurious OOD detections by real-time monitoring potential model misclassifications"

SHINE is a metric generated from five other metrics: "Similarity, hardness, i, noise, and entropy"; adapted to be applied during operation in safe-critical systems.

OOD data is usually source of errors in classifiers and object detectors built with deep learning.
Literature usually apply detectors at real-time aiming at detect and avoid such data.
However, not all OOD data exposed to the ML model will yield wrong predictions. 
To avoid spurious detections or spurious data processing at real-time, we developed this metric, which the goal is to give a clue of when an image (ID or OOD) is more error-prune to the model.
