# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 14:33:03 2023

@author: Kert PC
"""

import matplotlib.pyplot as plt

out_channels = [1, 2]
doubles = ['', '_double']


fig=plt.figure(figsize=(16, 16))
columns = 2
rows = 2

for i, out_channel in enumerate(out_channels) :
    for j, double in enumerate(doubles) :
        f_t = open(f"loss{out_channel}{double}_train.txt", "r")
        f_v = open(f"loss{out_channel}{double}_val.txt", "r")
        
        train_l = []
        val_l = []
        
        for loss in f_t :
            train_l.append(float(loss))
            
        for loss in f_v :
            val_l.append(float(loss))

        fig.add_subplot(rows, columns, i*columns + j + 1)
        plt.plot(train_l, label = 'training MSE')
        plt.plot(val_l, label = 'validation RMS')
        plt.title(f'Loss & RMS : ch{out_channel}{double}')
        plt.ylabel('value')
        plt.xlabel('epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')

plt.show()
                


