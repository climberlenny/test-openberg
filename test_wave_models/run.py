import pandas as pd
import numpy as np
from pprint import pprint
from time import time
from opendrift.models.openberg import OpenBerg
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
from datetime import timedelta
from toolbox.preprocessing import preprocessing
from toolbox.postprocessing import (
    postprocessing,
    compute_IDs,
    compute_SS,
    statistics,
)
from netCDF4 import Dataset
import os
import copernicusmarine
import pickle


input_folder = "DATA/FOR_LENNY/Observations/01_ICEPPR_drifter_data_cleaned"  # folder containing the csv files
output_folder = "test_openberg/test_wave_models/output"

with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]

USERNAME = username
PASSWORD = password

ocean_model = "cmems_obs_mob_glo_phy-cur_my_0.25deg_P1D-m"  # GLOB CURRENT
wind_model = "DATA/FOR_LENNY/WIND_MODELS/2021/ERA5/6h.10m_wind_2021.nc"  # ERA5
wave = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

columns = {0: "Latitude", 1: "Longitude", 2: "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%d-%m-%Y %H:%M",
    "time_thresh": 7,
    "dist_thresh": 20,
    "ts_interp": 600,
    "No_column": True,
}


def multiseeding(
    files: list[str],
    output_folder: str,
    prep_params: dict,
    Clean: bool = False,
    stokes=False,
    wave_rad=False,
    no_acc: bool = True,
    ts_calculation: int = 600,
    ts_observation: int = 600,
    ts_output: int = 3600,
):
    acc = "no_acc" if no_acc else "acc"
    SD = "SD" if stokes else "noSD"
    wave_model = "Wave" if wave_rad else "noWave"
    if not os.path.exists(
        os.path.join(output_folder, f"global_{SD}_{wave_model}_{acc}.nc")
    ):
        if no_acc:
            o = OpenBerg(loglevel=20)
        else:
            o = IcebergDrift(loglevel=20, with_stokes_drift=stokes, wave_rad=wave_rad)
        readers_current = copernicusmarine.open_dataset(
            dataset_id=ocean_model,
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

        reader_wind = reader_netCDF_CF_generic.Reader(wind_model)
        o.add_reader(reader_wind)

        readers_wave = copernicusmarine.open_dataset(
            dataset_id=wave,
            username=USERNAME,
            password=PASSWORD,
            # minimum_latitude=60,
        )
        readers_wave = reader_netCDF_CF_generic.Reader(
            readers_wave,
        )
        o.add_reader(readers_wave)

        o.set_config("environment:fallback:x_wind", 0)
        o.set_config("environment:fallback:y_wind", 0)
        o.set_config("drift:max_age_seconds", 86400 * 1 + 1)  # 1 days
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
                        o.seed_elements(
                            lon[k],
                            lat[k],
                            d,
                            z=0,
                        )
                        Day0 += timedelta(days=1)

        t1 = time()
        o.run(
            time_step=ts_calculation,
            steps=50000,
            time_step_output=ts_output,
            outfile=os.path.join(output_folder, f"global_{SD}_{wave_model}_{acc}.nc"),
            export_variables=["time", "age_seconds", "lon", "lat"],
        )
        t2 = time()
        print(t2 - t1)
        o.plot(fast=True)
    else:
        print("the output file already exists")
        if Clean:
            print("Replacing the file")
            os.remove(os.path.join(output_folder, f"global_{SD}_{wave_model}_{acc}.nc"))
            multiseeding(
                files=files,
                output_folder=output_folder,
                prep_params=prep_params,
                stokes=stokes,
                wave_rad=wave_rad,
                no_acc=no_acc,
                ts_calculation=ts_calculation,
                ts_observation=ts_observation,
                ts_output=ts_output,
            )

    return os.listdir(input_folder)


files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]
for bool_stokes in [False, True]:
    for bool_wave in [False, True]:
        multiseeding(
            files=files,
            output_folder=output_folder,
            prep_params=prep_params,
            stokes=bool_stokes,
            wave_rad=bool_wave,
            no_acc=False,
        )
multiseeding(
    files=files,
    output_folder=output_folder,
    prep_params=prep_params,
    stokes=False,
    wave_rad=False,
    no_acc=True,
)

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
            model = nc_f.removeprefix("global_").removesuffix(".nc")
            print(model)
            filename = os.path.join(output_folder, nc_f)
            Matches = postprocessing(
                [filename],
                input_folder,
                IDs,
                prep_params,
                outfile=os.path.join(output_folder, "list_obs.pickle"),
            )
            with open(os.path.join(output_folder, f"matches.pickle"), "wb") as f:
                pickle.dump(Matches, f)
            if len(Matches) > 0:
                data_SS = compute_SS(
                    Matches,
                    os.path.join(output_folder, "SS" + nc_f.replace(".nc", ".csv")),
                    save=True,
                )
                data_SS.loc[:, "wave model"] = model
                # statistics(
                #     data_SS,
                #     by=None,
                #     outfolder=os.path.join(
                #         output_folder, "stats/" + nc_f.replace(".nc", "/")
                #     ),
                # )
                df2save.append(data_SS)

    df2save = pd.concat(df2save, axis=0)
    df2save.to_csv(os.path.join(output_folder, "SS_2021.csv"), index=False)
else:
    df2save = pd.read_csv(os.path.join(output_folder, "SS_2021.csv"))
statistics(
    df2save, by=["wave model"], outfolder=os.path.join(output_folder, "stats/"), rot=90
)
