from datetime import datetime, timedelta
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
import os
from pprint import pprint
from toolbox.preprocessing import preprocessing
import pandas as pd
from netCDF4 import Dataset
from toolbox.postprocessing import (
    determine_contour_level,
)


wind_model = "DATA/FOR_LENNY/Wind/wind_ERA5_1990_4_7.nc"
ocean = os.listdir("DATA/FOR_LENNY/IK_data")

### READER
reader_wind = reader_netCDF_CF_generic.Reader(
    wind_model, standard_name_mapping={"u10": "x_wind", "v10": "y_wind"}
)
reader_current = reader_netCDF_CF_generic.Reader(
    [os.path.join("DATA/FOR_LENNY/IK_data", o) for o in ocean]
)
readers = [reader_current, reader_wind]
columns = {"LATITUDE": "Latitude", "LONGITUDE": "Longitude", "time": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 20,
    "ts_interp": 3600,
    "No_column": False,
}


def multiensembling(
    file,
    output_folder: str,
    prep_params: dict,
    readers,
    W: tuple,
    H: tuple,
    Ca: tuple,
    Co: tuple,
    distribution="normal",
    Clean: bool = False,
    ts_calculation: int = 900,
    ts_observation: int = 900,
    ts_output: int = 3600,
):
    W, Wstd = W
    H, Hstd = H
    minCa, maxCa, Ca, Ca_std = Ca
    minCo, maxCo, Co, Co_std = Co

    def call_opendrift(lon, lat, time, seed_num):
        o = IcebergDrift(
            loglevel=20,
            with_stokes_drift=True,
            wave_rad=False,
            grounding=True,
            melting=True,
        )
        o.add_reader(readers)
        o.set_config("environment:fallback:x_wind", 0)
        o.set_config("environment:fallback:y_wind", 0)
        o.set_config("drift:max_age_seconds", 86400 * 3 + 1)  # 3 days drift
        o.set_config("environment:fallback:x_sea_water_velocity", None)
        o.set_config("environment:fallback:y_sea_water_velocity", None)
        o.set_config("drift:max_speed", 10)

        o.seed_ensemble(
            (W, Wstd),
            (H, Hstd),
            (minCa, maxCa, Ca, Ca_std),
            (minCo, maxCo, Co, Co_std),
            lon=lon,
            lat=lat,
            time=time,
            z=0,
            number=10_000,
            drag_coeff_distribution=distribution,
        )
        o.run(
            time_step=ts_calculation,
            duration=timedelta(days=4),
            time_step_output=ts_output,
            outfile=os.path.join(output_folder, f"Day_{seed_num}.nc"),
            export_variables=["time", "age_seconds", "lon", "lat"],
        )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

        dfs = preprocessing(
            file,
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

            seed = 0
            Day0 = date[0] - timedelta(days=1)
            for k, d in enumerate(date):
                if d == Day0 + timedelta(days=1):
                    call_opendrift(lon=lon[k], lat=lat[k], time=d, seed_num=seed)
                    seed += 1

                    Day0 += timedelta(days=1)
    else:
        print("the output file already exists")
        if Clean:
            print("Replacing the file")
            os.remove(os.path.join(output_folder))
            multiensembling(
                file=file,
                output_folder=output_folder,
                prep_params=prep_params,
                readers=readers,
                W=(W, Wstd),
                H=(H, Hstd),
                Ca=(minCa, maxCa, Ca, Ca_std),
                Co=(minCo, maxCo, Co, Co_std),
                distribution=distribution,
                ts_calculation=ts_calculation,
                ts_observation=ts_observation,
                ts_output=ts_output,
            )

    return 0


ts = 900
ID = "7089"
file = os.path.join("test_openberg/test_IK/observations_Barents", f"{ID}.csv")
W = 90
H = 15 + 4 * 15  # see table 1 thesis IK
# Ensemble 1: normal (Ca=0.7 +/- 0.5, Co=0.25 +/- 0.2)

multiensembling(
    file,
    "test_openberg/test_IK/output/7089/Ensembles/multi_ensembles/Ensemble_1",
    prep_params,
    readers,
    W=(W, 20),
    H=(H, 20),
    Ca=(0.1, 4, 0.7, 0.5),
    Co=(0.1, 4, 0.25, 0.2),
    distribution="normal",
)
multiensembling(
    file,
    "test_openberg/test_IK/output/7089/Ensembles/multi_ensembles/Ensemble_2",
    prep_params,
    readers,
    W=(W, 20),
    H=(H, 20),
    Ca=(0.1, 4, 1, 0.5),
    Co=(0.1, 4, 1, 0.5),
    distribution="normal",
)
multiensembling(
    file,
    "test_openberg/test_IK/output/7089/Ensembles/multi_ensembles/Ensemble_3",
    prep_params,
    readers,
    W=(W, 20),
    H=(H, 20),
    Ca=(0.1, 4, 0.7, 0.5),
    Co=(0.1, 4, 0.25, 0.2),
    distribution="uniform",
)
multiensembling(
    file,
    "test_openberg/test_IK/output/7089/Ensembles/multi_ensembles/Ensemble_4",
    prep_params,
    readers,
    W=(W, 20),
    H=(H, 20),
    Ca=(0.1, 4, 1, 0.5),
    Co=(0.1, 4, 1, 0.5),
    distribution="uniform",
)
multiensembling(
    file,
    "test_openberg/test_IK/output/7089/Ensembles/multi_ensembles/Ensemble_5",
    prep_params,
    readers,
    W=(W, 20),
    H=(H, 20),
    Ca=(0.1, 4, 1.6, 1.5),
    Co=(0.1, 4, 1.6, 1.5),
    distribution="normal",
)


#### POSTPROCESS ###
ensemble_folder = (
    "test_openberg/test_IK/output/7089/Ensembles/multi_ensembles/Ensemble_3"
)

if not os.path.exists(os.path.join(ensemble_folder, "c02.csv")):
    dfs = preprocessing(
        file,
        column_names=prep_params["column_names"],
        date_format=prep_params["date_format"],
        time_thresh=prep_params["time_thresh"],
        dist_thresh=prep_params["dist_thresh"],
        timestep_interpolation=900,
        No_column=prep_params["No_column"],
    )

    C02 = {"level": [], "mu_b": [], "mu_r": [], "R": [], "e_x": [], "e_y": []}
    C04 = {"level": [], "mu_b": [], "mu_r": [], "R": [], "e_x": [], "e_y": []}
    C07 = {"level": [], "mu_b": [], "mu_r": [], "R": [], "e_x": [], "e_y": []}
    for df in dfs:
        lon = df.loc[:, "Longitude"].values
        lat = df.loc[:, "Latitude"].values
        date = df.index

        Day0 = date[0] - timedelta(days=1)
        day = 0
        for k, d in enumerate(date):
            if d == Day0 + timedelta(days=1):
                with Dataset(
                    os.path.join(ensemble_folder, f"Day_{day}.nc"), "r"
                ) as ensemble_data:
                    ens_lon = ensemble_data["lon"][:, :]
                    ens_lat = ensemble_data["lat"][:, :]
                end_date = d + pd.Timedelta(days=3)
                selected_data = df[(df.index >= d) & (df.index < end_date)]
                lon = selected_data["Longitude"].values
                lat = selected_data["Latitude"].values
                Day0 += timedelta(days=1)
                day += 1
                param1, disp = determine_contour_level(
                    None,
                    (lon, lat),
                    ensemble=(ens_lon, ens_lat),
                    c=0.2,
                )
                mu_b, mu_r, R, err = disp
                C02["level"].append(param1)
                C02["mu_b"].append(mu_b)
                C02["mu_r"].append(mu_r)
                C02["R"].append(R)
                C02["e_x"].append(err[0])
                C02["e_y"].append(err[1])
                param2, disp = determine_contour_level(
                    None,
                    (lon, lat),
                    ensemble=(ens_lon, ens_lat),
                    c=0.4,
                )
                mu_b, mu_r, R, err = disp
                C04["level"].append(param2)
                C04["mu_b"].append(mu_b)
                C04["mu_r"].append(mu_r)
                C04["R"].append(R)
                C04["e_x"].append(err[0])
                C04["e_y"].append(err[1])
                param3, disp = determine_contour_level(
                    None,
                    (lon, lat),
                    ensemble=(ens_lon, ens_lat),
                    c=0.7,
                )
                mu_b, mu_r, R, err = disp
                C07["level"].append(param3)
                C07["mu_b"].append(mu_b)
                C07["mu_r"].append(mu_r)
                C07["R"].append(R)
                C07["e_x"].append(err[0])
                C07["e_y"].append(err[1])

    data02 = pd.DataFrame(C02)
    data04 = pd.DataFrame(C04)
    data07 = pd.DataFrame(C07)
    data02.to_csv(os.path.join(ensemble_folder, "c02.csv"))
    data04.to_csv(os.path.join(ensemble_folder, "c04.csv"))
    data07.to_csv(os.path.join(ensemble_folder, "c07.csv"))
