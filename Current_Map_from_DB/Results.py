import os
from time import time
import numpy as np
import os
from toolbox.postprocessing import postprocessing, compute_IDs, polarplot2
from pprint import pprint
from netCDF4 import Dataset, num2date
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pickle
from opendrift.models.physics_methods import (
    skillscore_liu_weissberg,
)

output_folder = "test_openberg/Current_Map_from_DB/output"
input_folder = "DATA/FOR_LENNY/DB_csv"
folder_nc = "LENNY/6-march/output_nc"
output_csv = "LENNY/6-march/output_csv"

dffiles = pd.read_csv("LENNY/6-march/sorted_files.csv")
files = dffiles.loc[:, "name"].values

PATH = "test_openberg/Current_Map_from_DB"

columns = {"Latitude": "Latitude", "Longitude": "Longitude", "date": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 10000,
    "dist_thresh": -1000,
    "ts_interp": 3600,
    "No_column": False,
}


IDs = compute_IDs(
    files,
    input_folder=input_folder,
    outfile=os.path.join(PATH, "list_ID.pickle"),
    prep_params=prep_params,
)
print(IDs)

for nc_f in os.listdir(folder_nc):
    nc_f = os.path.join(folder_nc, nc_f)
    Matches = postprocessing(
        nc_f, input_folder, IDs, prep_params, ts_output=43200, files=files[:100]
    )
    if len(Matches) > 0:

        polarplot2(
            Matches,
            os.path.join(output_folder, f"polarplot.png"),
            # data_SS.loc[:, "SS"],
        )
    # pprint(Matches)
    stop
