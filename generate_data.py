# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:32:15 2023

@author: Kert PC
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import math

from datetime import date, timedelta
import matplotlib.pyplot as plt

def daterange(start_date, end_date):
    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)

DB_PATH = "data/years/"
TABLE_NAME = "measurements"
OUTPUT_PATH_IMG = "data/img/"
OUTPUT_PATH_NPY = "data/npy/"

eps = 1e-6

resolution = 0.25
lat_lower_lim = 30.0

lon_lower_lim = 7.0



for year in range(2020, 2021):
    con = sqlite3.connect(DB_PATH + f'{year}.db')
    
    
    start_date = date(year, 1, 1)
    end_date = date(year+1, 1, 1)
    for single_date in daterange(start_date, end_date):
        print(single_date.strftime("%Y-%m-%d"))
        
        end_time = (single_date + timedelta(1)).strftime("%Y-%m-%d") + ' 00:00'
        start_time = single_date.strftime("%Y-%m-%d") + ' 00:00'
        
        data = pd.read_sql(f"SELECT * FROM {TABLE_NAME} WHERE time < '{end_time}' AND time >= '{start_time}'", con)
        
        
        
        count = np.ones((44*4, 16*4)) * eps 
        bins = np.zeros((44*4, 16*4))
        
        for idx, point in data.iterrows() :
            if (point['latitude'] < 46.0 and point['latitude'] >= 30.0 
                and (point['longitude'] < 37.0 or point['longitude'] >= 353.0)
                and not math.isnan(point['sla_filtered'])) :
                   
                lat_idx = math.floor((point['latitude'] - lat_lower_lim) / resolution)
                
                if point['longitude'] < 37.0 :
                    lon_idx = math.floor((point['longitude'] + lon_lower_lim) / resolution)
                else : 
                    lon_idx = math.floor((point['longitude'] - 360.0 + lon_lower_lim) / resolution)
                    
                
                count[lon_idx, lat_idx] += 1
                bins[lon_idx, lat_idx] += point['sla_filtered']
                
        
        day_data = bins / count
        
        with open(OUTPUT_PATH_NPY + single_date.strftime("%Y-%m-%d") + '.npy', 'wb') as f :
            np.save(f, day_data)
            
        plt.imsave(OUTPUT_PATH_IMG + single_date.strftime("%Y-%m-%d") + '.png', day_data, cmap='gray')
        
    con.close()
        
