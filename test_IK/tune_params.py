from datetime import datetime, timedelta
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
import os
from pprint import pprint
from toolbox.preprocessing import preprocessing
from toolbox.postprocessing import tuning_drag_by_dev, tuning_drag_by_rmse
import pandas as pd
from opendrift.export.io_netcdf import import_file
import numpy as np
from netCDF4 import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from opendrift.models.physics_methods import distance_between_trajectories
import seaborn as sns


wind_model = "DATA/FOR_LENNY/Wind/wind_ERA5_1990_4_7.nc"
ocean = os.listdir("DATA/FOR_LENNY/IK_data/WSea_ice/")

### READER
reader_wind = reader_netCDF_CF_generic.Reader(
    wind_model, standard_name_mapping={"u10": "x_wind", "v10": "y_wind"}
)
reader_current = reader_netCDF_CF_generic.Reader(
    [os.path.join("DATA/FOR_LENNY/IK_data/WSea_ice/", o) for o in ocean],
    standard_name_mapping={
        "vxsi": "sea_ice_x_velocity",
        "vysi": "sea_ice_y_velocity",
    },
)
readers = [reader_current, reader_wind]


ts = 900
ID = "7088"
file = os.path.join("test_openberg/test_IK/observations_Barents", f"{ID}.csv")
with open(file) as f:
    data = pd.read_csv(f)
lon = data["LONGITUDE"][0]
lat = data["LATITUDE"][0]
start_time = pd.to_datetime(data["time"][0])
end_time = pd.to_datetime(data["time"].values[-1])

output_ID = os.path.join("test_openberg/test_IK/output/tuning", ID)

columns = {"LATITUDE": "Latitude", "LONGITUDE": "Longitude", "time": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 20,
    "ts_interp": 3600,
    "No_column": False,
}


def find_tuned_param(observation, ensemble, days, Co, Ca):
    i, theta, THETA = tuning_drag_by_dev(observation, ensemble, days)
    j, rmse, RMSE = tuning_drag_by_rmse(observation, ensemble)
    return (
        Ca[i],
        Co[i],
        theta,
        Ca[j],
        Co[j],
        rmse,
    )


def tuning_map(
    ensemble, observation, path, days, vmin=None, vmax=None, interp_method="nearest"
):
    nc_data = Dataset(ensemble)
    Co = nc_data["water_drag_coeff"][:, 0].data
    Ca = nc_data["wind_drag_coeff"][:, 0].data
    draft = nc_data["draft"][:, 0].data
    sail = nc_data["sail"][:, 0].data
    id_iceb = nc_data["trajectory"][:].data
    lon_mod = nc_data["lon"][:].data
    lat_mod = nc_data["lat"][:].data

    observation = pd.read_csv(observation)
    lon_obs = observation["LONGITUDE"].values[0 : 24 * days + 1]
    lat_obs = observation["LATITUDE"].values[0 : 24 * days + 1]

    valid = []
    for m in range(len(lon_mod)):
        if len(lon_mod[m, lon_mod[m] < 361]) == len(lon_obs):
            valid.append(m)
    RMSE = []
    RMSE2 = []
    for i in valid:
        dist = distance_between_trajectories(lon_obs, lat_obs, lon_mod[i], lat_mod[i])
        dist2 = distance_between_trajectories(
            lon_obs[-1:], lat_obs[-1:], lon_mod[i][-1:], lat_mod[i][-1:]
        )
        rmse = np.sqrt(np.nansum(dist**2) / len(dist))
        RMSE.append(rmse / 1000)
        RMSE2.append(dist2 / 1000)
    RMSE2 = np.concatenate(RMSE2)
    df = pd.DataFrame(
        {
            "Co": Co[valid],
            "Ca": Ca[valid],
            "draft": draft[valid],
            "sail": sail[valid],
            "ID": id_iceb[valid],
            "RMSE": RMSE,
            "RMSE2": RMSE2,
        }
    )
    df[["Co", "Ca"]] = np.round(df[["Co", "Ca"]], 3)
    # df.sort_values(by=["Co","Ca"])
    df = df.sort_values(by=["RMSE2", "RMSE"])
    # df = df.reset_index()
    pprint(df.iloc[0, :])
    index_dist = df.index[0]
    df = df.sort_values(by=["RMSE"])
    index_rmse = df.index[0]
    print(index_rmse, index_dist)
    min_rmse = df.loc[:, "RMSE"].min()
    min_rmse2 = df.loc[:, "RMSE2"].min()

    # Separate the data into Co, Ca, and THETA
    Co = df["Co"]
    Ca = df["Ca"]
    # THETA = df["THETA"]
    RMSE = df["RMSE"]
    RMSE2 = df["RMSE2"]

    # Define the grid for Co and Ca
    water_drag = np.linspace(np.min(Co), np.max(Co), 300)
    wind_drag = np.linspace(np.min(Ca), np.max(Ca), 300)
    water_drag, wind_drag = np.meshgrid(water_drag, wind_drag)

    # Interpolate THETA values onto the grid
    # theta_grid = griddata((Co, Ca), THETA, (water_drag, wind_drag), method='cubic')
    rmse_grid = griddata((Co, Ca), RMSE, (water_drag, wind_drag), method=interp_method)
    rmse_grid2 = griddata(
        (Co, Ca), RMSE2, (water_drag, wind_drag), method=interp_method
    )
    # Plot the grid using pcolormesh
    fig, ax = plt.subplots()
    cmap = sns.color_palette("rocket_r", as_cmap=True)

    c = ax.pcolormesh(
        water_drag,
        wind_drag,
        rmse_grid,
        cmap=cmap,
        vmin=vmin,
        vmax=(1 + vmax / 100) * min_rmse,
    )
    levels = [10, 20, 30, 40, 50, 60, 70, 80]
    contour = ax.contour(
        water_drag,
        wind_drag,
        rmse_grid,
        colors="#8fd400",
        levels=levels,
    )
    clabels = ax.clabel(
        contour,
        inline=True,
        fontsize=12,
        fmt={level: f"{int(level)}km" for level in levels},
    )
    cb = plt.colorbar(c, ax=ax, label="RMSE")
    cb.ax.tick_params(labelsize=12)
    cb.set_label("RMSE", fontsize=12)

    # Label the axes
    ax.set_xlabel("Water Drag (Co)", fontsize=12)
    ax.set_ylabel("Wind Drag (Ca)", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1)
    plt.savefig(f"{path}_TM.png", bbox_inches="tight", dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    c = ax.pcolormesh(
        water_drag,
        wind_drag,
        rmse_grid2,
        cmap=cmap,
        vmin=vmin,
        vmax=(1 + vmax / 100) * min_rmse2 + 10,
    )
    contour = ax.contour(
        water_drag,
        wind_drag,
        rmse_grid2,
        colors="#8fd400",
        levels=levels,
    )
    clabels = ax.clabel(
        contour,
        inline=True,
        fontsize=12,
        fmt={level: f"{int(level)}km" for level in levels},
    )
    cb = plt.colorbar(c, ax=ax, label="Dist to endpoint")
    cb.ax.tick_params(labelsize=12)
    cb.set_label("Dist to endpoint", fontsize=12)

    # Label the axes
    ax.set_xlabel("Water Drag (Co)", fontsize=12)
    ax.set_ylabel("Wind Drag (Ca)", fontsize=12)
    ax.tick_params(labelsize=12)
    ax.set_box_aspect(1)
    plt.savefig(f"{path}_TM2.png", bbox_inches="tight", dpi=300)
    plt.show()

    fig, ax = plt.subplots(
        figsize=(5, 5),
        subplot_kw={"projection": ccrs.NorthPolarStereo(central_longitude=32)},
    )

    # Configure gridlines
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
    )
    gl.top_labels = True
    gl.bottom_labels = False
    gl.left_labels = True
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter(zero_direction_label=False)
    gl.yformatter = LatitudeFormatter()
    gl.xlocator = mticker.MaxNLocator(3)
    gl.ylocator = mticker.MaxNLocator(3)
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    # Add features to the map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)

    colors = sns.color_palette("rocket")
    C1, C2 = df.loc[index_dist, "Ca"].round(2), df.loc[index_dist, "Co"].round(2)
    ax.plot(
        lon_mod[valid][index_dist, :],
        lat_mod[valid][index_dist, :],
        color=colors[2],
        transform=ccrs.PlateCarree(),
        label=f"best (dist to endpoint)\nCa={C1:.2f}, Co={C2:.2f}",
    )
    ax.scatter(
        lon_mod[valid][index_dist, :: 24 * 5],
        lat_mod[valid][index_dist, :: 24 * 5],
        marker="o",
        facecolor="white",
        color=colors[2],
        transform=ccrs.PlateCarree(),
    )
    C1, C2 = df.loc[index_rmse, "Ca"].round(2), df.loc[index_rmse, "Co"].round(2)
    ax.plot(
        lon_mod[valid][index_rmse, :],
        lat_mod[valid][index_rmse, :],
        color=colors[5],
        transform=ccrs.PlateCarree(),
        label=f"best (rmse)\nCa={C1:.2f}, Co={C2:.2f}",
    )
    ax.scatter(
        lon_mod[valid][index_rmse, :: 24 * 5],
        lat_mod[valid][index_rmse, :: 24 * 5],
        marker="o",
        facecolor="white",
        color=colors[5],
        transform=ccrs.PlateCarree(),
    )

    ax.plot(
        lon_obs,
        lat_obs,
        color=colors[0],
        transform=ccrs.PlateCarree(),
        label="observation",
    )
    ax.scatter(
        lon_obs[:: 24 * 5],
        lat_obs[:: 24 * 5],
        marker="o",
        facecolor="white",
        color=colors[0],
        transform=ccrs.PlateCarree(),
    )

    # Set the extent and aspect ratio
    ax.set_extent([30, 33, 76.5, 79.5], crs=ccrs.PlateCarree())
    ax.set_box_aspect(1)
    ax.legend(fontsize=12)
    plt.savefig(f"{path}_BF.png", bbox_inches="tight", dpi=300)
    plt.show()


# 7089 : L=95, W = 90, H(sail) = 15
# 7088 : L=95, W = 80, H(sail) = 20
# 7087 : L=63, W = 56, H(sail) = 10
# 7086 : L=90, W = 60, H(sail) = 10

W = 80
H = 20 + 4 * 20  # see table 1 thesis IK
Ca = 0.7
Co = 0.25


name = "Ensemble_CaCo_30_M4.nc"
if not os.path.exists(os.path.join(output_ID, name)):
    o = IcebergDrift(
        loglevel=20,
        add_stokes_drift=False,
        wave_rad=False,
        grounding=False,
        melting=True,
        vertical_profile=False,
        add_coriolis=True,
    )
    o.set_config("drift:max_speed", 10)
    o.set_config("drift:truncate_ocean_model_below_m", 1000)
    o.add_reader(readers)
    # linearly_spaced_values = np.linspace(10, 101, 10)
    # height = np.repeat(linearly_spaced_values, 100)
    height = 100
    draft = 4 / 5 * height
    sail = 1 / 5 * height
    o.seed_ensemble(
        (1.2, 1.2),
        (1.2, 1.2),
        lon=lon,
        lat=lat,
        width=80,
        length=95,
        draft=draft,
        sail=sail,
        time=start_time,
        drag_coeff_distribution="uniform",
        number=10000,
    )

    o.run(
        duration=timedelta(days=30),
        time_step=ts,
        time_step_output=3600,
        outfile=os.path.join(output_ID, name),
    )
    # o.plot(fast=True)

tuning_map(
    "test_openberg/test_IK/output/tuning/7088/Ensemble_CaCo_30_M_C_VP.nc",
    "test_openberg/test_IK/observations_Barents/7088.csv",
    "test_openberg/test_IK/output/tuning/7088/Ensemble_CaCo_30_M_C_VP",
    30,
    vmax=40,
    interp_method="linear",
)
# tuning_map(
#     "test_openberg/test_IK/output/debug/7088/drift_30_7.nc",
#     "test_openberg/test_IK/observations_Barents/7088.csv",
#     "test_openberg/test_IK/output/debug/7088/tuning_map_30_M",
#     30,
#     vmax=60,
# )


# days = 30
# lon_obs = data["LONGITUDE"].values[0 : 24 * days + 1].reshape((-1, 1))
# lat_obs = data["LATITUDE"].values[0 : 24 * days + 1].reshape((-1, 1))
# observation = np.hstack((lon_obs, lat_obs))
# print(observation.shape)
# with Dataset("test_openberg/test_IK/output/tuning/7089/Ensemble_CaCo_30.nc") as nc_data:
#     lon_mod = nc_data["lon"][:].data
#     lat_mod = nc_data["lat"][:].data
#     Co = nc_data["water_drag_coeff"][:, 0].data
#     Ca = nc_data["wind_drag_coeff"][:, 0].data
#     members = []
#     valid = []
#     for m in range(len(lon_mod)):
#         lon_m = lon_mod[m]
#         lon_m[lon_m > 360] = lon_m[lon_m < 360][-1]
#         lat_m = lat_mod[m]
#         lat_m[lat_m > 360] = lat_m[lat_m < 360][-1]
#         lon_m = lon_m[:].reshape((-1, 1))
#         lat_m = lat_m[:].reshape((-1, 1))
#         if len(lon_m[lon_m < 361]) == len(observation):
#             valid.append(m)
#             member = np.hstack((lon_m, lat_m))
#             members.append(member.reshape(1, -1, 2))
#     Co = Co[valid]
#     Ca = Ca[valid]
#     ensemble = np.vstack(members)
# print(ensemble.shape)
# print(find_tuned_param(observation, ensemble, days, Co, Ca))

# Create a figure and axis with NorthPolarStereo projection
