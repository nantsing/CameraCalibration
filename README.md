# CameraCalibration

## files
1. dataset: Contains pictures used in experiments.
2. results: Contains results of 2 method.
3. rectification.py: Reproduction of *Computing Rectifying Homographies for Stereo Vi-
sion* from [github@*decadenza*](https://github.com/decadenza/DirectStereoRectification).
4. Transform.py: Use API in rectification.py to make a demo.
5. rectify.py: Reproduction of *A compact algorithm for rectification of stereo pairs*.

## Usage:
1. To get rectifications of left.png and right.png:

        python rectify.py

2. To get rectifications of 1.png and 2.png:

         python Transform.py