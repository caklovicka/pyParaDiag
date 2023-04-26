#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 2023

@author: gaya
"""
import numpy as np
from pySDC.core.Collocation import CollBase
import matplotlib.pyplot as plt

# change these:
############################
M = 2
quadType = 'RADAU-RIGHT'
distr = 'LEGENDRE'

lIm1 = -15
lIm2 = 15
lRe1 = -10
lRe2 = 15

############################

def f():
    # l0 = implict
    # l1 = explicit

    r = np.empty((lr.shape[0], li.shape[0]))
    for i in range(li.shape[0]):
        for j in range(lr.shape[0]):
            R = np.linalg.inv(np.eye(M) - (lr[j] + 1j * li[i]) * Q) @ Q
            r[j, i] = np.linalg.norm(R, np.inf)
    return r

def g():
    # l0 = implict
    # l1 = explicit

    r = np.empty((lr.shape[0], li.shape[0]))
    for i in range(li.shape[0]):
        for j in range(lr.shape[0]):
            r[j, i] = 1 / (1 - np.abs(lr[j] + 1j * li[i]))
    return r

def h():
    r = np.empty((lr.shape[0], li.shape[0]))
    for i in range(li.shape[0]):
        for j in range(lr.shape[0]):
            r[j, i] = 1 / np.abs(lr[j] + 1j * li[i])
    return r


coll = CollBase(M, 0, 1, node_type='LEGENDRE', quad_type='RADAU-RIGHT')
Q = coll.Qmat[1:, 1:]
lr = np.array(np.linspace(lIm1, lIm2, 300))
li = np.array(np.linspace(lRe1, lRe2, 300))
x = np.meshgrid(lr, li)
f()

plt.rcParams['contour.negative_linestyle'] = 'solid'
fig, ax = plt.subplots()
CSf = ax.contour(li, lr, f(), 6, colors='k', levels=[0.2, 0.5, 1])  # Negative contours default to dashed.
#CSg = ax.contour(li, lr, g(), 6, colors='r', levels=[0.5, 1])  # Negative contours default to dashed.
#CSh = ax.contour(li, lr, h(), 6, colors='g', levels=[0.5, 1])  # Negative contours default to dashed.
ax.clabel(CSf, fontsize=12, inline=True)
#ax.clabel(CSg, fontsize=9, inline=True)
#ax.clabel(CSh, fontsize=9, inline=True)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.xlabel(r'$Re\lambda_I$', fontsize=20)
plt.ylabel(r'$Im\lambda_I$', fontsize=20)
plt.tight_layout()
plt.show()


