from opendrift.readers import reader_netCDF_CF_generic
from opendrift.models.openberg import OpenBerg
from opendrift.models.openberg_acc import IcebergDrift
from datetime import datetime, timedelta
import copernicusmarine
import os
from toolbox.preprocessing import preprocessing
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]

USERNAME = username
PASSWORD = password

# test 1
lon1 = -56.391907
lat = 72.082367
start_time = datetime(2021, 9, 22, 14, 32)

wind_model = "DATA/FOR_LENNY/WIND_MODELS/2021/ERA5/6h.10m_wind_2021.nc"
GLORYS = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
wave = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

### READER
reader_wind = reader_netCDF_CF_generic.Reader(wind_model)
reader_current = copernicusmarine.open_dataset(
    dataset_id=GLORYS, username=USERNAME, password=PASSWORD
)
reader_current = reader_netCDF_CF_generic.Reader(reader_current)
reader_wave = copernicusmarine.open_dataset(
    dataset_id=wave, username=USERNAME, password=PASSWORD
)
reader_wave = reader_netCDF_CF_generic.Reader(reader_wave)

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "no_acc_model.nc")):
    o1 = OpenBerg(loglevel=0)
    o1.add_reader([reader_wind, reader_current])

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "model_acc.nc")):
    o2 = IcebergDrift(loglevel=0, correction=False)  # with acceleration only
    o2.add_reader([reader_wind, reader_current, reader_wave])

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "model_acc_SD.nc")):
    o3 = IcebergDrift(
        loglevel=0, wave_model="SD", correction=False
    )  # with Stokes drift only
    o3.add_reader([reader_wind, reader_current, reader_wave])

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "model_acc_RF.nc")):
    o4 = IcebergDrift(
        loglevel=0, wave_model="RF", correction=False
    )  # with Wave radiation Force only
    o4.add_reader([reader_wind, reader_current, reader_wave])

if not os.path.exists(
    os.path.join("test_openberg/test_acc/output", "model_acc_SDRF.nc")
):
    o5 = IcebergDrift(
        loglevel=0, wave_model="SDRF", correction=False
    )  # with Stokes drift and Wave radiation Force
    o5.add_reader([reader_wind, reader_current, reader_wave])

ts = 600
if not os.path.exists(os.path.join("test_openberg/test_acc/output", "no_acc_model.nc")):
    o1.seed_elements(lon1, lat, number=1, time=start_time)
    o1.run(
        duration=timedelta(days=4),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join("test_openberg/test_acc/output", "no_acc_model.nc"),
    )
else:
    o1 = OpenBerg()
    o1.io_import_file(os.path.join("test_openberg/test_acc/output", "no_acc_model.nc"))

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "model_acc.nc")):
    o2.seed_elements(lon1, lat, number=1, time=start_time, derivation_timestep=ts)
    o2.run(
        duration=timedelta(days=4),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join("test_openberg/test_acc/output", "model_acc.nc"),
    )
else:
    o2 = IcebergDrift()
    o2.io_import_file(os.path.join("test_openberg/test_acc/output", "no_acc_model.nc"))

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "model_acc_SD.nc")):
    o3.seed_elements(lon1, lat, number=1, time=start_time, derivation_timestep=ts)
    o3.run(
        duration=timedelta(days=4),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join("test_openberg/test_acc/output", "model_acc_SD.nc"),
    )
else:
    o3 = IcebergDrift(wave_model="SD")
    o3.io_import_file(os.path.join("test_openberg/test_acc/output", "model_acc_SD.nc"))

if not os.path.exists(os.path.join("test_openberg/test_acc/output", "model_acc_RF.nc")):
    o4.seed_elements(lon1, lat, number=1, time=start_time, derivation_timestep=ts)
    o4.run(
        duration=timedelta(days=4),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join("test_openberg/test_acc/output", "model_acc_RF.nc"),
    )
else:
    o4 = IcebergDrift(wave_model="RF")
    o4.io_import_file(os.path.join("test_openberg/test_acc/output", "model_acc_RF.nc"))

if not os.path.exists(
    os.path.join("test_openberg/test_acc/output", "model_acc_SDRF.nc")
):
    o5.seed_elements(lon1, lat, number=1, time=start_time, derivation_timestep=ts)
    o5.run(
        duration=timedelta(days=4),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join("test_openberg/test_acc/output", "model_acc_SDRF.nc"),
    )
else:
    o5 = IcebergDrift(wave_model="SDRF")
    o5.io_import_file(
        os.path.join("test_openberg/test_acc/output", "model_acc_SDRF.nc")
    )

# Plot results
# o1.plot(fast=True, compare=o3)

columns = {0: "Latitude", 1: "Longitude", 2: "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%d-%m-%Y %H:%M",
    "time_thresh": 7,
    "dist_thresh": 0,
    "ts_interp": 600,
    "No_column": True,
}

obs = preprocessing(
    "DATA/FOR_LENNY/Observations/01_ICEPPR_drifter_data_cleaned/11_DSE_300434065758620.csv",
    column_names=prep_params["column_names"],
    date_format=prep_params["date_format"],
    time_thresh=prep_params["time_thresh"],
    dist_thresh=prep_params["dist_thresh"],
    timestep_interpolation=prep_params["ts_interp"],
    No_column=prep_params["No_column"],
)
model_no_acc = Dataset(os.path.join("test_openberg/test_acc/output", "no_acc_model.nc"))
model_acc = Dataset(os.path.join("test_openberg/test_acc/output", "model_acc.nc"))
model_acc_SD = Dataset(os.path.join("test_openberg/test_acc/output", "model_acc_SD.nc"))
model_acc_RF = Dataset(os.path.join("test_openberg/test_acc/output", "model_acc_RF.nc"))
model_acc_SDRF = Dataset(
    os.path.join("test_openberg/test_acc/output", "model_acc_SDRF.nc")
)


def model_check(obs, model1, model2, model3=None, model4=None, model5=None):
    lon_obs = obs.loc[:, "Longitude"].values
    lat_obs = obs.loc[:, "Latitude"].values

    lon1 = model1["lon"][0].data
    lat1 = model1["lat"][0].data

    lon2 = model2["lon"][0].data
    lat2 = model2["lat"][0].data

    if not model3 is None:
        lon3 = model3["lon"][0].data
        lat3 = model3["lat"][0].data
    else:
        lon3 = None
        lat3 = None

    if not model4 is None:
        lon4 = model4["lon"][0].data
        lat4 = model4["lat"][0].data
    else:
        lon4 = None
        lat4 = None

    if not model5 is None:
        lon5 = model5["lon"][0].data
        lat5 = model5["lat"][0].data
    else:
        lon5 = None
        lat5 = None

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree(-55.5)})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
    )
    gl.right_labels = False
    ax.plot(lon_obs, lat_obs, transform=ccrs.PlateCarree(), color="k", label="obs")
    ax.plot(lon1, lat1, transform=ccrs.PlateCarree(), color="r", label="no_acc_model")
    ax.plot(
        lon2,
        lat2,
        transform=ccrs.PlateCarree(),
        color="g",
        linestyle="--",
        label="model_acc",
    )
    ax.plot(
        lon3,
        lat3,
        transform=ccrs.PlateCarree(),
        color="b",
        linestyle="-.",
        label="model_acc_SD",
    )
    ax.plot(
        lon4,
        lat4,
        transform=ccrs.PlateCarree(),
        color="purple",
        linestyle=":",
        label="model_acc_RF",
    )
    ax.plot(
        lon5,
        lat5,
        transform=ccrs.PlateCarree(),
        color="orange",
        linestyle="-",
        label="model_acc_SDRF",
    )

    plt.legend()
    plt.show()


model_check(obs[0], model_no_acc, model_acc, model_acc_SD, model_acc_RF, model_acc_SDRF)
