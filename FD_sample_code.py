#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 18:30:46 2025

@author: jschlieffen
"""

import numpy as np

def compute_gradient(f, x, y, z):
    grad = np.zeros((len(x), len(y), len(z), 3)) 
    
    for i in range(1, len(x) - 1):
        for j in range(len(y)):
            for k in range(len(z)):
                grad[i, j, k, 0] = (f[i+1, j, k] - f[i-1, j, k]) / (x[i+1] - x[i-1])

    for i in range(len(x)):
        for j in range(1, len(y) - 1):
            for k in range(len(z)):
                grad[i, j, k, 1] = (f[i, j+1, k] - f[i, j-1, k]) / (y[j+1] - y[j-1])

    for i in range(len(x)):
        for j in range(len(y)):
            for k in range(1, len(z) - 1):
                grad[i, j, k, 2] = (f[i, j, k+1] - f[i, j, k-1]) / (z[k+1] - z[k-1])

    return grad


x = np.array([0, 1, 2.5, 5, 8])
y = np.array([0, 0.8, 2, 3.5, 5])
z = np.array([0, 0.5, 2, 3.3, 4, 5])

f = np.sin(x[:, None, None]) * np.cos(y[None, :, None]) * np.exp(-z[None, None, :])

grad = compute_gradient(f, x, y, z)

print("Gradient at (i=2, j=2, k=2):", grad[2, 2, 2])
