import os
import numpy as np

# IntrinsicMatrix
K = np.array([
    [525.0, 0, 319.5],
    [0, 525.0, 239.5],
    [0, 0, 1],
])

# 
def q2R(q):
    x, y, z, w = q
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
    
    return R
    
def f2qt(f, q, t):
    for i, item in enumerate(f):
        if i < 3: t.append(float(item))
        else: q.append(float(item))
    
def Transform(K1, R1, t1, K2, R2, t2):
    c1 = -R1.T @ K1.inv() @ t1
    c2 = -R2.T @ K1.inv() @ t1
    


if __name__ == '__main__':
    
    q1 = []
    t1 = []
    q2 = []
    t2 = []
    with open('camera.txt', 'r') as file:
        files = file.read().split('\n')
        f1 = files[0].split(' ')
        f2 = files[1].split(' ')
    f2qt(f1, q1, t1)
    f2qt(f2, q2, t2)
    R1 = q2R(q1)
    R2 = q2R(q2)
    T1, T2 = Transform(K, R1, t1, K, R2, t2)    