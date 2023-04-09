import os
import numpy as np

# IntrinsicMatrix
K = np.array([
    [525.0, 0, 319.5],
    [0, 525.0, 239.5],
    [0, 0, 1],
])

def q2R(x, y, z, w):
    r1 = 1 - 2 * y**2 - 2 * z**2
    r2 = 2 * x * y - 2 * w * z 
    r3 = 2 * x * z + 2 * w * y
    r4 = 2 * x * y + 2 * w * z
    r5 = 1 - 2 * x**2 - 2 * z**2
    r6 = 2 * y * z - 2 * w * x
    r7 = 2 * x * z - 2 * w * y
    r8 = 2 * y * z + 2 * w * x
    r9 = 1 - 2 * x**2 - 2 * y**2
    R = np.array([
        [r1, r2, r3],
        [r4, r5, r6],
        [r7, r8, r9],
    ])


with open('camera.txt', 'r') as file:
    print(file.read().split('\n'))
    