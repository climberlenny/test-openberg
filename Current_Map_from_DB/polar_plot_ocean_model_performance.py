import copernicusmarine
import numpy as np
import pandas as pd
from toolbox.preprocessing import preprocessing, Nearest2DInterpolator
import os
import pyproj
from pprint import pprint

dirname = "DATA/FOR_LENNY/DB_csv2"  # drifting buoys trajectories > 50 deg N

ocean_models = {
    "GLOB": [
        "cmems_obs_mob_glo_phy-cur_my_0.25deg_P1D-m",
        "cmems_obs_mob_glo_phy-cur_nrt_0.25deg_P1D-m",
    ],  # after 2022
    "GLORYS": [
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",  # before 30/06/2021
        "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
    ],  # after 01/07/2021
    "TOPAZ4": ["cmems_mod_arc_phy_my_topaz4_P1D-m"],  # to 31/12/2022
}

columns = {"Latitude": "Latitude", "Longitude": "Longitude", "date": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 0,
    "ts_interp": 3600,
    "No_column": False,
}
with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]

USERNAME = username
PASSWORD = password

for file in os.listdir(dirname):
    file = os.path.join(dirname, file)
    observations = preprocessing(
        file,
        column_names=columns,
        date_format=prep_params["date_format"],
        time_thresh=prep_params["time_thresh"],
        dist_thresh=prep_params["dist_thresh"],
        timestep_interpolation=prep_params["ts_interp"],
        No_column=prep_params["No_column"],
    )
    for obs in observations:
        geod = pyproj.Geod(ellps="WGS84")
        lon = obs.loc[:, "Longitude"].values
        lat = obs.loc[:, "Latitude"].values
        start = obs.index[0]
        end = obs.index[-1]
        if start.year > 2022:
            continue
        # observation = ref
        azimuth_alpha, a2, distance0 = geod.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
        vel0 = distance0 / 3600
        pprint(vel0)
        readers_current = copernicusmarine.open_dataset(
            dataset_id=ocean_models["GLOB"][0],
            username=USERNAME,
            password=PASSWORD,
            minimum_latitude=50,
            start_datetime=start,
            end_datetime=end,
            variables=["uo", "vo"],
            minimum_depth=-10,
        )
        pprint(readers_current)
        stop
