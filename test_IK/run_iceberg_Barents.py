from datetime import datetime, timedelta
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
import os
from pprint import pprint
from toolbox.preprocessing import preprocessing
import pandas as pd
from opendrift.export.io_netcdf import import_file


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


# 7089 : L=95, W = 90, H(sail) = 15
# 7088 : L=95, W = 80, H(sail) = 20
# 7087 : L=63, W = 56, H(sail) = 10
# 7086 : L=90, W = 60, H(sail) = 10

ts = 900
# test 1: full drift
ID = "7088"
file = os.path.join("test_openberg/test_IK/observations_Barents", f"{ID}.csv")
with open(file) as f:
    data = pd.read_csv(f)
lon = data["LONGITUDE"][0]
lat = data["LATITUDE"][0]
start_time = pd.to_datetime(data["time"][0])
end_time = pd.to_datetime(data["time"].values[-1])


o = IcebergDrift(
    loglevel=20,
    add_stokes_drift=True,
    wave_rad=False,
    grounding=True,
    melting=True,
    vertical_profile=True,
)
o.add_reader(readers)
o.seed_elements(
    lon,
    lat,
    start_time,
    wind_drag_coeff=0.7,
    water_drag_coeff=0.25,
    length=95,
    width=80,
    sail=20,
    draft=4 * 20,
)
# o.seed_elements(lon, lat, start_time)
output_ID = os.path.join("test_openberg/test_IK/output", ID)
if not os.path.exists(output_ID):
    os.mkdir(output_ID)
if not os.path.exists(os.path.join(output_ID, "drift_30_070025.nc")):
    o.run(
        end_time=start_time + timedelta(days=30),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join(output_ID, "drift_30_070025.nc"),
    )
    o.plot(filename=os.path.join(output_ID, "drift_30_070025.png"))

# test 2 multiseeding


def multiseeding(
    file,
    output_folder: str,
    prep_params: dict,
    readers,
    Clean: bool = False,
    ts_calculation: int = 900,
    ts_observation: int = 900,
    ts_output: int = 3600,
):
    if not os.path.exists(os.path.join(output_folder, "Multiseeding.nc")):
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
        o.set_config("drift:max_age_seconds", 86401)
        o.set_config("environment:fallback:x_sea_water_velocity", None)
        o.set_config("environment:fallback:y_sea_water_velocity", None)

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

            Day0 = date[0] - timedelta(days=1)
            for k, d in enumerate(date):
                if d == Day0 + timedelta(days=1):
                    o.seed_elements(lon[k], lat[k], d, z=0)
                    Day0 += timedelta(days=1)
        o.run(
            time_step=ts_calculation,
            steps=50000,
            time_step_output=ts_output,
            outfile=os.path.join(output_folder, "Multiseeding.nc"),
            export_variables=["time", "age_seconds", "lon", "lat"],
        )
        o.plot(fast=True)
    else:
        print("the output file already exists")
        if Clean:
            print("Replacing the file")
            os.remove(os.path.join(output_folder, "Multiseeding.nc"))
            multiseeding(
                file=file,
                output_folder=output_folder,
                prep_params=prep_params,
                readers=readers,
                ts_calculation=ts_calculation,
                ts_observation=ts_observation,
                ts_output=ts_output,
            )

    return 0


columns = {"LATITUDE": "Latitude", "LONGITUDE": "Longitude", "time": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 20,
    "ts_interp": 3600,
    "No_column": False,
}
# multiseeding(
#     file=file,
#     output_folder=output_ID,
#     prep_params=prep_params,
#     readers=readers,
#     ts_observation=3600,
#     ts_output=6 * 3600,
# )
multiseeding(
    file=file,
    output_folder=output_ID,
    prep_params=prep_params,
    readers=readers,
)


# test 3 : evolution of the score
def multiseeding2(
    file,
    output_folder: str,
    prep_params: dict,
    readers,
    Clean: bool = False,
    ts_calculation: int = 900,
    ts_observation: int = 900,
    ts_output: int = 3600,
):
    if not os.path.exists(os.path.join(output_folder, "Multiseeding_evolution_SS.nc")):
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
        o.set_config("drift:max_age_seconds", 86400 * 10 + 1)
        o.set_config("environment:fallback:x_sea_water_velocity", None)
        o.set_config("environment:fallback:y_sea_water_velocity", None)

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

            Day0 = date[0] - timedelta(days=5)
            for k, d in enumerate(date):
                if d == Day0 + timedelta(days=5):
                    o.seed_elements(lon[k], lat[k], d, z=0)
                    Day0 += timedelta(days=5)
        o.run(
            time_step=ts_calculation,
            steps=50000,
            time_step_output=ts_output,
            outfile=os.path.join(output_folder, "Multiseeding_evolution_SS.nc"),
            export_variables=["time", "age_seconds", "lon", "lat"],
        )
        o.plot(fast=True)
    else:
        print("the output file already exists")
        if Clean:
            print("Replacing the file")
            os.remove(os.path.join(output_folder, "Multiseeding_evolution_SS.nc"))
            multiseeding2(
                file=file,
                output_folder=output_folder,
                prep_params=prep_params,
                readers=readers,
                ts_calculation=ts_calculation,
                ts_observation=ts_observation,
                ts_output=ts_output,
            )

    return 0


multiseeding2(
    file=file,
    output_folder=output_ID,
    prep_params=prep_params,
    readers=readers,
    ts_observation=3600,
    ts_output=6 * 3600,
)


# test4 with ensemble 3,5,10,30 jours (W,H)


W = 90
H = 15 + 4 * 15  # see table 1 thesis IK
Ca = 0.7
Co = 0.25
# 3days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HW_3.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 4)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(100, 100, 1, 1, 1),
#     )

#     o.run(
#         duration=timedelta(days=3),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HW_3.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HW_3.mp4"))
# # 5days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HW_5.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 4)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(100, 100, 1, 1, 1),
#     )

#     o.run(
#         duration=timedelta(days=5),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HW_5.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HW_5.mp4"))
# # 10days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HW_10.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 4)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(100, 100, 1, 1, 1),
#     )

#     o.run(
#         duration=timedelta(days=10),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HW_10.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HW_10.mp4"))

# # 30days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HW_30.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 4)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(100, 100, 1, 1, 1),
#     )

#     o.run(
#         duration=timedelta(days=30),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HW_30.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HW_30.mp4"))

# # test5 with ensemble 3,5,10,30 jours (Ca,Co)

# W = 90
# H = 15 + 4 * 15  # see table 1 thesis IK
# Ca = 0.7
# Co = 0.25
# # 3days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_3.nc")):
#     o = IcebergDrift(loglevel=0, with_stokes_drift=True, wave_rad=False, grounding=True)
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 1),
#         (0.25, 1),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(1, 1, 100, 100, 1),
#     )

#     o.run(
#         duration=timedelta(days=3),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_3.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_3.mp4"))
# # 5days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_5.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 1),
#         (0.25, 1),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(1, 1, 100, 100, 1),
#     )

#     o.run(
#         duration=timedelta(days=5),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_5.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_5.mp4"))
# # 10days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_10.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 1),
#         (0.25, 1),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(1, 1, 100, 100, 1),
#     )

#     o.run(
#         duration=timedelta(days=10),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_10.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_10.mp4"))

# # 30days
if not os.path.exists(
    os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_30_melting.nc")
):
    o = IcebergDrift(
        loglevel=20,
        add_stokes_drift=True,
        wave_rad=False,
        grounding=True,
        melting=True,
    )
    o.set_config("drift:max_speed", 10)
    o.add_reader(readers)
    o.seed_ensemble(
        (W, 1),
        (H, 1),
        (0.01, 5, 0.7, 1),
        (0.01, 5, 0.25, 1),
        lon=lon,
        lat=lat,
        time=start_time,
        numbers=(1, 1, 100, 100, 1),
        number=10000,
    )

    o.run(
        duration=timedelta(days=30),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_30_melting.nc"),
    )
    o.plot(fast=True)
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_CaCo_30.mp4"))

# # test6 with ensemble 3,5,10,30 jours (H,W,Ca,Co)

# W = 90
# H = 15 + 4 * 15  # see table 1 thesis IK
# Ca = 0.7
# Co = 0.25
# # 3days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble6_HWCaCo_3.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 20),
#         (H, 20),
#         (0.1, 4, 1.6, 1.5),
#         (0.1, 4, 1.6, 1.5),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         number=10_000,
#         drag_coeff_distribution="uniform",
#     )

#     o.run(
#         duration=timedelta(days=3),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble6_HWCaCo_3.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble6_HWCaCo_3.gif"))
# # 5days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_5.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 1),
#         (0.25, 1),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(10, 10, 10, 10, 1),
#     )

#     o.run(
#         duration=timedelta(days=5),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_5.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_5.mp4"))
# # 10days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_10.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 1),
#         (0.25, 1),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(10, 10, 10, 10, 1),
#     )

#     o.run(
#         duration=timedelta(days=10),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_10.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_10.mp4"))

# # 30days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_30.nc")):
#     o = IcebergDrift(loglevel=0, with_stokes_drift=True, wave_rad=False, grounding=True)
#     o.set_config("drift:max_speed", 10)
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 400),
#         (H, 400),
#         (0.7, 1),
#         (0.25, 1),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(10, 10, 10, 10, 1),
#     )
#     o.run(
#         duration=timedelta(days=30),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_30.nc"),
#     )
#     o.animation(filename=os.path.join(output_ID, "Ensembles", "Ensemble_HWCaCo_30.mp4"))

# # test7 with ensemble 3,5,10,30 jours (radius=1000m)

# W = 90
# H = 15 + 4 * 15  # see table 1 thesis IK
# Ca = 0.7
# Co = 0.25
# # 3days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_1000m_3.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(1, 1, 1, 1, 1000),
#         radius=1000,
#     )

#     o.run(
#         duration=timedelta(days=3),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_1000m_3.nc"),
#     )
#     # o.animation(filename=os.path.join(output_ID, "Ensemble_1000m_3_anim.mp4"))
# # 5days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_1000m_5.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(1, 1, 1, 1, 1000),
#         radius=1000,
#     )

#     o.run(
#         duration=timedelta(days=5),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_1000m_5.nc"),
#     )
# # 10days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_1000m_10.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         radius=1000,
#         numbers=(1, 1, 1, 1, 1000),
#     )

#     o.run(
#         duration=timedelta(days=10),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_1000m_10.nc"),
#     )

# # 30days
# if not os.path.exists(os.path.join(output_ID, "Ensembles", "Ensemble_1000m_30.nc")):
#     o = IcebergDrift(
#         loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True
#     )
#     o.add_reader(readers)
#     o.seed_ensemble(
#         (W, 1),
#         (H, 1),
#         (0.7, 0.01),
#         (0.25, 0.01),
#         lon=lon,
#         lat=lat,
#         time=start_time,
#         numbers=(1, 1, 1, 1, 1000),
#         radius=1000,
#     )

#     o.run(
#         duration=timedelta(days=30),
#         time_step=ts,
#         time_step_output=3600,
#         outfile=os.path.join(output_ID, "Ensembles", "Ensemble_1000m_30.nc"),
#     )


# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_3.gif"
# ):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_3.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_3.gif",
#         fast=True,
#         colorbar=False,
#         show_trajectories=True,
#         trajectory_alpha=0.01,
#         fps=20,
#     )

# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_5.gif"
# ):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_5.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_5.gif",
#         fast=True,
#         colorbar=False,
#         show_trajectories=True,
#         trajectory_alpha=0.01,
#         fps=40,
#     )

# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_10.mp4"
# ):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_10.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_10.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_30.gif"
# ):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_30.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_30.gif",
#         fast=False,
#         show_trajectories=True,
#         trajectory_alpha=0.01,
#         fps=40,
#     )
# if not os.path.exists("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_3.mp4"):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_3.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_3.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_5.mp4"):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_5.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_5.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_10.mp4"):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_10.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_10.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_30.gif"):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_30.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HW_30.gif",
#         fast=True,
#         corners=[24.5, 42.5, 69, 80],
#         show_trajectories=True,
#         trajectory_alpha=0.01,
#         fps=40,
#     )
# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_3.mp4"
# ):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_3.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_3.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_5.mp4"
# ):
#     o = IcebergDrift()
#     o.io_import_file("test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_5.nc")
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_5.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_10.mp4"
# ):
#     o = IcebergDrift()
#     o.io_import_file(
#         "test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_10.nc"
#     )
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_10.mp4",
#         fast=True,
#         colorbar=False,
#         fps=20,
#     )
# if not os.path.exists(
#     "test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_30.gif"
# ):
#     o = IcebergDrift()
#     o.io_import_file(
#         "test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_30.nc"
#     )
#     # o.plot(fast=True)
#     o.animation(
#         filename="test_openberg/test_IK/output/7089/Ensembles/Ensemble_HWCaCo_30.gif",
#         fast=True,
#         show_trajectories=True,
#         trajectory_alpha=0.01,
#         fps=40,
#     )
