# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 11:46:04 2023

@author: Kert PC
"""

import os
import sqlite3
import pandas as pd

DB_PATH = "data/store.db"
TABLE_NAME = "measurements"
OUTPUT_PATH = "data/years"
TIME_NAME = "time"

con = sqlite3.connect(DB_PATH)

for year in range(1992, 2010):
    print(f"Year: {year}")
    save_con = sqlite3.connect(os.path.join(OUTPUT_PATH, f"{year}.db"))
    
    end_time = f'{year+1}-01-01 00:00'
    start_time = f'{year}-01-01 00:00'
    
    data = pd.read_sql(f"SELECT * FROM {TABLE_NAME} WHERE time < '{end_time}' AND time >= '{start_time}'", con)
    data.to_sql(TABLE_NAME, save_con, index=False)
    save_con.close()
