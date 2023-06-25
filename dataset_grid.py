# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:17:19 2023

@author: Kert PC
"""

import os
import torch
import torch.utils.data as data
import random
import pandas as pd
import numpy as np
import math
from datetime import date


class SatDataset(data.Dataset):

    def __init__(self, path, t_window, test=False) -> None:
        super().__init__()
        self.path = path
        self.test = test
        self.t_window = t_window
        self.files = os.listdir(path)
       
        self.indices = np.arange(t_window // 2, len(self.files) - (t_window // 2), 1)
        #random.shuffle(self.indices)
        
        step = 0.25 / 180
        self.longitude = np.arange(-7/180, 37/180, step)
        self.latitude = np.arange(30/180, 46/180, step)
        
        self.longitude = np.expand_dims(self.longitude, axis=-1) @ np.ones((self.latitude.size, 1)).T
        self.latitude = np.ones((self.longitude.shape[0], 1)) @ np.expand_dims(self.latitude, axis=-1).T
        

    def __len__(self):
        return len(self.indices)
    
    def get_time_encoding(self, date_string='2010-01-01'):
        split = date_string.split('-')
        year = int(split[0])
        month = int(split[1])
        day = int(split[2])
        
        d0 = date(year, 1, 1)
        d1 = date(year, month, day)
        diff = d0 - d1
        
        cnst = math.pi / 365.25
        no_days = diff.days
        
        sine = math.sin(no_days * cnst)
        cosine = math.cos(no_days * cnst) 
        
        sin_arr = np.full(self.longitude.shape, sine)
        cos_arr = np.full(self.longitude.shape, cosine)
        
        return sin_arr, cos_arr
        

    def __getitem__(self, index) :
        file_index = self.indices[index]
        curr_date = self.files[file_index]
        date = curr_date.split('.')[0]
        sin_arr, cos_arr = self.get_time_encoding(date)
        
        item_list = [self.longitude, self.latitude, sin_arr, cos_arr]
        
        gt = np.load(self.path + curr_date)
        mask = torch.from_numpy(gt != 0)
        
        for neighbor_day in self.files[file_index - (self.t_window // 2) : file_index + (self.t_window // 2) + 1] :
            day_data = np.load(self.path + neighbor_day)
            if neighbor_day == curr_date :
                day_data = np.zeros(day_data.shape)
            item_list.append(day_data)
            
        item = np.stack(item_list, axis=0)
        
        return {'satData' : item, 'gt' : gt, 'mask' : mask, 'date' : curr_date}
    



if __name__ == "__main__":
    st = SatDataset('data/npy/', 9)
    st.__getitem__(0)
    
    
    
