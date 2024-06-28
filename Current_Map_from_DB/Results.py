import os
from toolbox.postprocessing import (
    postprocessing,
    compute_IDs,
    compute_SS,
    statistics,
    polarplot2,
    polarplot_contour,
    plot_current_map,
)
from pprint import pprint
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.patches import FancyBboxPatch


output_folder = "test_openberg/Current_Map_from_DB/output3"
input_folder = "DATA/FOR_LENNY/DB_csv2"

dffiles = pd.read_csv("test_openberg/Current_Map_from_DB/sorted_files.csv")
files = dffiles.loc[:, "files"].values

PATH = "test_openberg/Current_Map_from_DB"

columns = {"Latitude": "Latitude", "Longitude": "Longitude", "date": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 0,
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


group = list(range(0, len(files), 100))
group.append(None)

models = ["TOPAZ4"]
for model in models:
    SS = []
    nc_f = [
        os.path.join(output_folder, f"global_{model}_{i}_run.nc") for i in group[0:-1]
    ]
    if os.path.exists(os.path.join(output_folder, f"{model}_matches.pickle")):
        with open(os.path.join(output_folder, f"{model}_matches.pickle"), "rb") as f:
            Matches = pickle.load(f)
    else:
        Matches = postprocessing(
            nc_f,
            input_folder,
            IDs,
            prep_params,
            ts_output=43200,
            files=files,
            outfile=os.path.join(
                "test_openberg/Current_Map_from_DB", "list_obs.pickle"
            ),
        )
        with open(os.path.join(output_folder, f"{model}_matches.pickle"), "wb") as f:
            pickle.dump(Matches, f)
    if len(Matches) > 0:
        SS_data = compute_SS(
            Matches,
            outfile=os.path.join(output_folder, f"SS_{model}.csv"),
            save=True,
        )
        SS.append(SS_data)
    polarplot_contour(
        Matches, os.path.join(output_folder, f"{model}_polarplot.png"), c=0.1
    )
    # pprint(Matches)
    # SS = pd.concat(SS)
    # SS.to_csv(os.path.join(output_folder, f"SS_{model}.csv"),index=False)

SS = pd.read_csv("test_openberg/Current_Map_from_DB/output3/SS_TOPAZ4.csv")
plot_current_map(SS, "test_openberg/Current_Map_from_DB/output3/", "TOPAZ4")
# statistics(SS, by=None, outfolder="test_openberg/Current_Map_from_DB/output")
