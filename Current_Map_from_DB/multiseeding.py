import os
from time import time
from datetime import timedelta
from opendrift.models.openberg import OpenBerg
from opendrift.readers import reader_netCDF_CF_generic
import os
from toolbox.preprocessing import preprocessing
from pprint import pprint
import copernicusmarine
import pandas as pd


output_folder = "test_openberg/Current_Map_from_DB/output"

input_folder = "DATA/FOR_LENNY/DB_csv2"


# sorting the csv files by date
if not os.path.exists(
    os.path.join("test_openberg/Current_Map_from_DB", "sorted_files.csv")
):
    columns = {"Latitude": "Latitude", "Longitude": "Longitude", "date": "time"}
    prep_params = {
        "column_names": columns,
        "date_format": "%Y-%m-%d %H:%M:%S",
        "time_thresh": 10000,
        "dist_thresh": 0,
        "ts_interp": 86400,
        "No_column": False,
    }

    df = {
        "files": [],
        "year1": [],
        "year2": [],
    }
    for file in os.listdir(input_folder):
        fp = os.path.join(input_folder, file)
        DATA = preprocessing(
            fp,
            column_names=prep_params["column_names"],
            date_format=prep_params["date_format"],
            time_thresh=prep_params["time_thresh"],
            dist_thresh=prep_params["dist_thresh"],
            timestep_interpolation=prep_params["ts_interp"],
            Cut=False,
        )
        for data in DATA:
            year1 = data.index.min().year
            year2 = data.index.max().year
            df["files"].append(file)
            df["year1"].append(year1)
            df["year2"].append(year2)

    df = pd.DataFrame(df)
    df = df.sort_values(by=["year1", "year2"])
    df = df.drop_duplicates(subset=["files"])
    df.to_csv(
        os.path.join("test_openberg/Current_Map_from_DB", "sorted_files.csv"),
        index=False,
    )
dffiles = pd.read_csv("test_openberg/Current_Map_from_DB/sorted_files.csv")
files = dffiles.loc[:, "files"].values

# Copernicus id and password
with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]
USERNAME = username
PASSWORD = password

ocean_models = {
    "GLOB": [
        "cmems_obs_mob_glo_phy-cur_my_0.25deg_P1D-m",
        "cmems_obs_mob_glo_phy-cur_nrt_0.25deg_P1D-m",  # after 2022
    ],
    "GLORYS": [
        "cmems_mod_glo_phy_my_0.083deg_P1D-m",  # before 30/06/2021
        "cmems_mod_glo_phy_myint_0.083deg_P1D-m",
    ],  # after 01/07/2021
    "TOPAZ4": ["cmems_mod_arc_phy_my_topaz4_P1D-m"],  # to 31/12/2022
}


def multiseeding(
    files: list[str],
    output_folder: str,
    prep_params: dict,
    Clean: bool = False,
    ocean_model: str = None,
    ts_calculation: int = 900,
    ts_observation: int = 900,
    ts_output: int = 3600,
):
    group = list(range(0, len(files), 100))
    group.append(None)
    for i, j in zip(group[0:-1], group[1:]):
        print(i, j)
        if not os.path.exists(
            os.path.join(output_folder, f"global_{ocean_model}_{i}_run.nc")
        ):
            o = OpenBerg(loglevel=20)
            for om in ocean_models[ocean_model]:
                readers_current = copernicusmarine.open_dataset(
                    dataset_id=om,
                    username=USERNAME,
                    password=PASSWORD,
                    # minimum_latitude=60,
                )
                readers_current = reader_netCDF_CF_generic.Reader(
                    readers_current,
                )
                o.add_reader(readers_current)

            o.set_config("environment:fallback:x_wind", 0)
            o.set_config("environment:fallback:y_wind", 0)
            o.set_config("drift:max_age_seconds", 86401)
            o.set_config("environment:fallback:x_sea_water_velocity", None)
            o.set_config("environment:fallback:y_sea_water_velocity", None)

            for fp in files[i:j]:
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
                outfile=os.path.join(output_folder, f"global_{ocean_model}_{i}_run.nc"),
                export_variables=["time", "age_seconds", "lon", "lat"],
            )
            t2 = time()
            print(t2 - t1)
            # o.plot(fast=True)
        else:
            print("the output file already exists")
            if Clean:
                print("Replacing the file")
                os.remove(
                    os.path.join(output_folder, f"global_{ocean_model}_{i}_run.nc")
                )
                multiseeding(
                    files=files,
                    output_folder=output_folder,
                    prep_params=prep_params,
                    ocean_model=ocean_model,
                    ts_calculation=ts_calculation,
                    ts_observation=ts_observation,
                    ts_output=ts_output,
                )

    return os.listdir(input_folder)


columns = {"Latitude": "Latitude", "Longitude": "Longitude", "date": "time"}
prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 0,
    "ts_interp": 3600,
    "No_column": False,
}
files = [os.path.join(input_folder, file) for file in files]
for ocean_model in ocean_models.keys():
    print(ocean_model)
    multiseeding(
        files=files,
        output_folder=output_folder,
        prep_params=prep_params,
        ocean_model=ocean_model,
        ts_calculation=3600,
        ts_observation=3600,
        ts_output=43200,
    )
