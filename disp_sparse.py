# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:21:04 2023

@author: Kert PC
"""

import numpy as np
import matplotlib.pyplot as plt


MASK_PATH = './data/mask/mask.npy'
READING_PATH = './data/npy/train/2016-12-06.npy'

mask = np.load(MASK_PATH)
mask_disp = np.rot90(mask)

reading = np.load(READING_PATH)
reading = np.rot90(reading)

fig=plt.figure(figsize=(12, 28))

plt.title(f'2016-12-06')
ssh = plt.imshow(reading, cmap='gray')
plt.imshow(mask_disp, cmap='gray', alpha=1.0*(mask_disp == 0), interpolation='none')