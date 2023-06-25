# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:38:41 2023

@author: Kert PC
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import math

from datetime import date, timedelta
import matplotlib.pyplot as plt


data = np.load('test.npy')

data = (data != 0)

plt.imsave('mask.png', data, cmap='gray')

with open('mask.npy', 'wb') as f :
    np.save(f, data)