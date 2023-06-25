import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
from netCDF4 import Dataset

DATA_FOLDER = "./data/original/"
DB_FILE = "store.db"
TABLE_NAME = "measurements"

def netcdf_to_dataframe(filepath: str) -> pd.DataFrame:
    """ Open given .nc file and put its variables in a DataFrame. """
    rootgrp = Dataset(filepath, "r+")
    df = pd.DataFrame()
    for var in rootgrp.variables.keys():
        df[var] = rootgrp.variables[var][:]
    if "time" in df.columns:
        df.loc[:, "time"] = pd.to_timedelta(df.loc[:, "time"], unit="days")
        df.loc[:, "time"] = pd.to_datetime(df.loc[:, "time"] + datetime(1950,1,1,0,0,0,0))
    rootgrp.close()
    return df

def create_db(db_file, table_name):
    if os.path.exists(db_file):
        os.remove(db_file)

    con = sqlite3.connect(db_file)

    i = 0
    for filepath in Path(DATA_FOLDER).glob("**/*.nc"):
        i += 1
        #if i == 100 :
        #    break
        df = netcdf_to_dataframe(filepath)
        df["mission"] = filepath.stem.split("_")[2]
        affected_rows = df.to_sql(table_name, con=con, if_exists="append", index=False)
        if i % 1000 == 0:
            print(f"Read {i} files.")
        #assert affected_rows == df.shape[0]
    print(f"Completed! Total files read: {i}")

if __name__ == "__main__":
    create_db(DB_FILE, TABLE_NAME)
