import pandas as pd
import numpy as np
from pprint import pprint
from time import time
from opendrift.models.openberg import OpenBerg
from opendrift.readers import reader_netCDF_CF_generic
from datetime import timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from toolbox.preprocessing import preprocessing
from toolbox.postprocessing import (
    postprocessing,
    compute_IDs,
    compute_SS,
    polarplot,
    statistics,
)
from netCDF4 import Dataset
import os
import copernicusmarine
import pickle


input_folder = "DATA/FOR_LENNY/Observations/01_ICEPPR_drifter_data_cleaned"  # folder containing the csv files
output_folder = "test_openberg/Model_Benchmark/output"

with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]

USERNAME = username
PASSWORD = password

ocean_models = {
    "GLOB": ["cmems_obs_mob_glo_phy-cur_my_0.25deg_P1D-m"],
    "GLORYS": [
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",  # avant le 30/06/2021
        "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
    ],  # après le 01/07/2021
    "TOPAZ4": ["cmems_mod_arc_phy_my_topaz4_P1D-m"],
    # "TOPAZ5": ["cmems_mod_arc_phy_anfc_6km_detided_P1D-m"],
    # "TOPAZ6": ["dataset-topaz6-arc-15min-3km-be"],
}

wind_models = {
    "ERA5": [
        "DATA/FOR_LENNY/WIND_MODELS/2011/ERA5/6h.10m_wind_2011.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2012/ERA5/6h.10m_wind_2012.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2013/ERA5/6h.10m_wind_2013.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2021/ERA5/6h.10m_wind_2021.nc",
    ],
    "CARRA": [
        "DATA/FOR_LENNY/WIND_MODELS/2011/CARRA/param_165.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2011/CARRA/param_166.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2012/CARRA/param_165.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2012/CARRA/param_166.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2013/CARRA/param_165.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2013/CARRA/param_166.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2021/CARRA/param_165.nc",
        "DATA/FOR_LENNY/WIND_MODELS/2021/CARRA/param_166.nc",
    ],
    None: [],
}


columns = {0: "Latitude", 1: "Longitude", 2: "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%d-%m-%Y %H:%M",
    "time_thresh": 7,
    "dist_thresh": 20,
    "ts_interp": 900,
    "No_column": True,
}


def multiseeding(
    files: list[str],
    output_folder: str,
    prep_params: dict,
    Clean: bool = False,
    ocean_model: str = None,
    wind_model: str = None,
    ts_calculation: int = 900,
    ts_observation: int = 900,
    ts_output: int = 3600,
):
    if not os.path.exists(
        os.path.join(output_folder, f"global_{ocean_model}_{wind_model}_run.nc")
    ):
        o = OpenBerg(loglevel=0)
        for om in ocean_models[ocean_model]:
            readers_current = copernicusmarine.open_dataset(
                dataset_id=om,
                username=USERNAME,
                password=PASSWORD,
                # minimum_latitude=60,
            )
            readers_current = reader_netCDF_CF_generic.Reader(
                readers_current,
                standard_name_mapping={
                    "eastward_sea_water_velocity": "x_sea_water_velocity",
                    "northward_sea_water_velocity": "y_sea_water_velocity",
                },
            )
            o.add_reader(readers_current)

        if not wind_model is None:
            reader_wind = reader_netCDF_CF_generic.Reader(wind_models[wind_model])
            o.add_reader(reader_wind)

        o.set_config("environment:fallback:x_wind", 0)
        o.set_config("environment:fallback:y_wind", 0)
        o.set_config("drift:max_age_seconds", 86401)
        o.set_config("environment:fallback:x_sea_water_velocity", None)
        o.set_config("environment:fallback:y_sea_water_velocity", None)

        for fp in files:
            dfs = preprocessing(
                fp,
                column_names=prep_params["column_names"],
                date_format=prep_params["date_format"],
                time_thresh=prep_params["time_thresh"],
                dist_thresh=prep_params["dist_thresh"],
                timestep_interpolation=ts_observation,
                No_column=prep_params["No_column"],
            )
            for df in dfs:
                lon = df.loc[:, "Longitude"].values
                lat = df.loc[:, "Latitude"].values
                date = df.index

                Day0 = date[0] - timedelta(days=1)
                for k, d in enumerate(date):
                    if d == Day0 + timedelta(days=1):
                        o.seed_elements(lon[k], lat[k], d, z=0)
                        Day0 += timedelta(days=1)

        t1 = time()
        o.run(
            time_step=ts_calculation,
            steps=50000,
            time_step_output=ts_output,
            outfile=os.path.join(
                output_folder, f"global_{ocean_model}_{wind_model}_run.nc"
            ),
            export_variables=["time", "age_seconds", "lon", "lat"],
        )
        t2 = time()
        print(t2 - t1)
        o.plot(fast=True)
    else:
        print("the output file already exists")
        if Clean:
            print("Replacing the file")
            os.remove(
                os.path.join(output_folder, f"global_{ocean_model}_{wind_model}_run.nc")
            )
            multiseeding(
                files=files,
                output_folder=output_folder,
                prep_params=prep_params,
                ocean_model=ocean_model,
                wind_model=wind_model,
                ts_calculation=ts_calculation,
                ts_observation=ts_observation,
                ts_output=ts_output,
            )

    return os.listdir(input_folder)


files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
for wind_model in wind_models.keys():
    for ocean_model in ocean_models.keys():
        print(ocean_model, wind_model)
        multiseeding(
            files=files,
            output_folder=output_folder,
            prep_params=prep_params,
            ocean_model=ocean_model,
            wind_model=wind_model,
        )

nc_f = "test_openberg/Model_Benchmark/output/global_GLOB_ERA5_run.nc"

IDs = compute_IDs(
    os.listdir(input_folder),
    input_folder,
    os.path.join(output_folder, "list_ID.pickle"),
    prep_params=prep_params,
)
pprint(IDs)

if not os.path.exists(os.path.join(output_folder, "SS_2021.csv")):
    df2save = []
    for nc_f in os.listdir(output_folder):
        if ".nc" in nc_f:
            dummy = nc_f.split("_")
            ocean_model = dummy[1]
            wind_model = dummy[2]
            if ocean_model == "GLOB":
                ocean_model = "GLOB CURRENT"
            if wind_model == "None":
                wind_model = "No wind"

            print(wind_model, ocean_model)
            nc_f = os.path.join(output_folder, nc_f)
            Matches = postprocessing(nc_f, input_folder, IDs, prep_params)
            if len(Matches) > 0:
                data_SS = compute_SS(
                    Matches, os.path.join(output_folder, "SS_2021.csv")
                )
                data_SS.loc[:, "ocean model"] = ocean_model
                data_SS.loc[:, "wind model"] = wind_model
                df2save.append(data_SS)

    df2save = pd.concat(df2save, axis=0)
    df2save.to_csv(os.path.join(output_folder, "SS_2021.csv"), index=False)
else:
    df2save = pd.read_csv(os.path.join(output_folder, "SS_2021.csv"))
statistics(
    df2save, outfolder=os.path.join("test_openberg/Model_Benchmark/output", "stats")
)
# polarplot(
#     Matches, os.path.join(output_folder, "polarplot.png")
# )
