from datetime import datetime, timedelta
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import pandas as pd
import netCDF4 as nc
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker
import os
import pickle
from toolbox.postprocessing import (
    plot_ensemble_mean,
    plot_contour_ensemble,
    postprocessing_1,
    compute_IDs,
    polarplot2,
    compute_SS,
    statistics,
    determine_contour_level,
    polarplot_contour,
)
from opendrift.models.physics_methods import (
    skillscore_liu_weissberg,
    distance_between_trajectories,
)

mpl.rcParams["font.size"] = 12


    """Code very useful to plot ensemble distribution
    """

def compute_score(lon, lat, lon_mod, lat_mod):
    SS = []
    RMSE = []
    max_index = min(len(lon), len(lon_mod[0]))
    lon = lon[:max_index]
    lat = lat[:max_index]
    lon_mod = lon_mod[0, :max_index]
    lat_mod = lat_mod[0, :max_index]
    for i in range(2, len(lon)):
        ss = skillscore_liu_weissberg(lon[:i], lat[:i], lon_mod[:i], lat_mod[:i])
        SS.append(ss)
        dist = distance_between_trajectories(lon[:i], lat[:i], lon_mod[:i], lat_mod[:i])
        rmse = np.sqrt(np.nansum(dist**2)) / len(dist)
        RMSE.append(rmse)

    return SS, RMSE


def plot_full_drift(lon_obs, lat_obs, lon_mod, lat_mod, time, projection):
    fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(12, 10))

    # Add coastlines and land features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")

    # Customize gridlines
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="dotted",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True

    # Plot the observed data
    ax.plot(
        lon_obs,
        lat_obs,
        transform=ccrs.PlateCarree(),
        color="blue",
        label="Observation Track",
    )
    ax.scatter(
        lon_obs[:: 24 * 10],
        lat_obs[:: 24 * 10],
        transform=ccrs.PlateCarree(),
        color="blue",
        edgecolor="black",
        zorder=5,
        label="positions every 10 days",
        s=50,
    )

    # Plot the model data
    ax.plot(
        lon_mod[0],
        lat_mod[0],
        transform=ccrs.PlateCarree(),
        color="orange",
        linestyle="--",
        label="Model Track",
    )
    # ax.scatter(
    #     lon_mod[:, :: 24 * 10],
    #     lat_mod[:, :: 24 * 10],
    #     transform=ccrs.PlateCarree(),
    #     color="orange",
    #     edgecolor="black",
    #     zorder=5,
    #     label="10 days positions model",
    #     s=50,
    # )
    ax.scatter(
        lon_mod[:, -1],
        lat_mod[:, -1],
        marker="o",
        edgecolor="k",
        facecolor="white",
        alpha=0.7,
        transform=ccrs.PlateCarree(),
    )

    # Set the extent to focus on the area of interest
    ax.set_extent([28, 40, 74, 80], crs=ccrs.PlateCarree())

    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:3], labels[:3], loc="lower left", fontsize=14)

    # Add a title
    # plt.title("Observation and Model Tracks", fontsize=16)

    # Show the plot
    return fig


def plot_score_evol(matches):
    SS = []
    for match in matches:
        obs, model = match[0], match[1]
        lon_obs, lat_obs = obs[:, 0], obs[:, 1]
        lon_mod, lat_mod = model[:, 0], model[:, 1]
        SS.append(compute_score(lon_obs, lat_obs, lon_mod, lat_mod))
    # print(len(SS), SS)
    fig, axs = plt.subplots(4, 3, figsize=(15, 10), sharex=True, sharey=True)
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            ax = axs[i, j]
            index = i * 3 + j
            if index < len(SS):
                ax.plot(
                    SS[index],
                    label=f"from day {5*(index)}",
                )
                ax.hlines(
                    np.mean(SS[index]),
                    0,
                    len(SS[index]),
                    color="r",
                    linestyle="--",
                    label="mean",
                )
                if j == 0:
                    ax.set_ylabel("score")
                num_points = len(SS[index])
                ticks = range(0, num_points, 8)
                labels = [f"Day {(k * 2)}" for k in range(len(ticks))]

                ax.set_xticks(ticks)
                ax.set_xticklabels(labels, rotation=45)
                ax.tick_params(direction="in")
                ax.legend(loc="lower right")
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=1.0)
    return fig


ID = 7088
with open(
    f"C:/Users/lenuch/ACCIBERG/test_openberg/test_IK/observations_Barents/{ID}.csv", "r"
) as f:
    df = pd.read_csv(f)
df["time"] = pd.to_datetime(df["time"])
selected_data = df[df["time"].dt.month < 6]
lon = df["LONGITUDE"].values
lat = df["LATITUDE"].values

with nc.Dataset(f"test_openberg/test_IK/output/debug/{ID}/drift_30_7.nc", "r") as data:
    lon_model = data["lon"][:]
    lat_model = data["lat"][:]
    wind_speed = np.sqrt(data["x_wind"][:] ** 2 + data["y_wind"][:] ** 2)
    ocean_speed = np.sqrt(
        data["x_sea_water_velocity"][:] ** 2 + data["y_sea_water_velocity"][:] ** 2
    )
    time = data["time"]
    time = (time - time[0]) / 3600


columns = {"LATITUDE": "Latitude", "LONGITUDE": "Longitude", "time": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 20,
    "ts_interp": 3600,
    "No_column": False,
}
# nc_file = f"test_openberg/test_IK/output/{ID}/Multiseeding.nc"
input_folder = "test_openberg/test_IK/observations_Barents/"
obs_file = f"{ID}.csv"
projection = ccrs.NorthPolarStereo(34)
# fig = plot_full_drift(lon, lat, lon_model, lat_model, time, projection)
# plt.savefig(
#     f"test_openberg/test_IK/output/debug/{ID}/drift_30_7.png",
#     bbox_inches="tight",
# )
# stop
# IDs = compute_IDs(
#     [f"{ID}.csv"],
#     "test_openberg/test_IK/observations_Barents/",
#     f"test_openberg/test_IK/output/{ID}/list_ID.pickle",
#     prep_params,
# )
# matches = postprocessing_1(
#     nc_file,
#     "test_openberg/test_IK/observations_Barents/",
#     [obs_file],
#     IDs,
#     prep_params,
#     outfile=f"test_openberg/test_IK/output/{ID}/list_obs.pickle",
#     ts_output=6 * 3600,
# )
# with open(
#     os.path.join(f"test_openberg/test_IK/output/{ID}/multiseeding_matches.pickle"),
#     "wb",
# ) as f:
#     pickle.dump(matches, f)


# polarplot_contour(matches, f"test_openberg/test_IK/output/{ID}/polarplot.png")
# SS = compute_SS(matches, f"test_openberg/test_IK/output/{ID}/SS.csv", save=True)
# statistics(SS, by=None, outfolder=f"test_openberg/test_IK/output/{ID}/stats")

# score, rmse = compute_score(lon, lat, lon_model, lat_model)
# rmse = np.array(rmse) / 1000
# print(len(rmse[: len(time) + 1 : 24]), len(time[2::24]))
# fig, ax = plt.subplots(figsize=(8, 3))
# handles = []
# line_score = ax.plot(
#     time[2 : min(len(lon), len(lon_model[0]))], score, label="Relative score"
# )
# score_1 = ax.scatter(
#     time[2:50:3], score[:48:3], marker="+", color="k", label="score every 3h"
# )
# score_2 = ax.scatter(
#     time[2:50:12],
#     score[:48:12],
#     marker="x",
#     color="r",
#     zorder=10,
#     s=100,
#     label="score every 12h",
# )
# ax.set_xlabel("time(h)")
# ax.set_ylabel("score")

# ax2 = ax.twinx()
# line_rmse = ax2.plot(
#     time[2 : min(len(lon), len(lon_model[0]))],
#     rmse,
#     color="b",
#     linestyle="-.",
#     label="RMSE",
# )
# rmse_1 = ax2.scatter(
#     time[2 : 1902 : 10 * 24],
#     rmse[: 1900 : 10 * 24],
#     marker="o",
#     edgecolor="k",
#     color="white",
#     alpha=0.5,
#     label="rmse every 10 days",
# )
# handles, labels = ax.get_legend_handles_labels()
# handles2, labels2 = ax2.get_legend_handles_labels()
# handles, labels = handles + handles2, labels + labels2
# ax2.set_ylabel("rmse (km)")
# plt.legend(handles=handles, labels=labels, loc="lower right")
# plt.savefig(
#     f"test_openberg/test_IK/output/{ID}/SS_evol.png",
#     bbox_inches="tight",
# )
# plt.show()


# nc_file = f"test_openberg/test_IK/output/{ID}/Multiseeding_evolution_SS.nc"
# input_folder = "test_openberg/test_IK/observations_Barents/"
# obs_file = f"{ID}.csv"
# projection = ccrs.NorthPolarStereo(34)
# projection = ccrs.PlateCarree(34)
# plot_full_drift(lon, lat, lon_model, lat_model, time, projection)
# IDs = compute_IDs(
#     [f"{ID}.csv"],
#     "test_openberg/test_IK/observations_Barents/",
#     f"test_openberg/test_IK/output/{ID}/list_ID2.pickle",
#     prep_params,
# )
# matches10 = postprocessing_1(
#     nc_file,
#     "test_openberg/test_IK/observations_Barents/",
#     [obs_file],
#     IDs,
#     prep_params,
#     outfile=f"test_openberg/test_IK/output/{ID}/list_obs.pickle",
#     ts_output=6 * 3600,
#     days=10,
# )

# SS = compute_SS(matches10, f"test_openberg/test_IK/output/{ID}/SS_10days.csv", save=True)
# statistics(SS, by=None, outfolder=f"test_openberg/test_IK/output/{ID}/stats_10days")
# fig = plot_score_evol(matches10)
# plt.savefig(
#     f"test_openberg/test_IK/output/{ID}/evol_score.png",
#     bbox_inches="tight",
# )
# plt.show()


name = "Ensemble_CaCo_30_M"
ensemble_file = f"test_openberg/test_IK/output/tuning/{ID}/{name}.nc"
# ensemble_file = f"test_openberg/test_IK/output/7089/Ensembles/Ensemble_CaCo_30.nc"
start_date = df["time"].min()
end_date = start_date + pd.Timedelta(days=30)
selected_data = df[(df["time"] >= start_date) & (df["time"] < end_date)]
lon = selected_data["LONGITUDE"].values
lat = selected_data["LATITUDE"].values

fig = plt.figure(1, figsize=(12, 4.5))
gs = gridspec.GridSpec(5, 6)
gs.update(wspace=0.1, hspace=0.1)
xtr_subplot = fig.add_subplot(gs[:5, :2], projection=projection)
ax1, contourf1, contour1 = plot_contour_ensemble(
    ensemble_file,
    observation=(lon, lat),
    c=0.2,
    grid_size=100,
    projection=projection,
    ax=xtr_subplot,
)
gl = ax1.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.top_labels = True
gl.bottom_labels = False
gl.left_label = True
gl.right_labels = False
gl.xformater = LongitudeFormatter(zero_direction_label=False)
gl.yformater = LatitudeFormatter(direction_label=False)
gl.xlocator = mticker.MaxNLocator(3)
gl.ylocator = mticker.MaxNLocator(3)
gl.xlabel_style = {"size": 10}
gl.ylabel_style = {"size": 10}

xtr_subplot = fig.add_subplot(gs[:5, 2:4], projection=projection)
ax2, contourf2, contour2 = plot_contour_ensemble(
    ensemble_file,
    observation=(lon, lat),
    c=0.4,
    grid_size=100,
    projection=projection,
    ax=xtr_subplot,
)
gl = ax2.gridlines(
    draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.top_labels = True
gl.bottom_labels = False
gl.left_label = False
gl.right_labels = False
gl.xformater = LongitudeFormatter(zero_direction_label=False)
gl.yformater = LatitudeFormatter(direction_label=False)
gl.xlocator = mticker.MaxNLocator(3)
gl.ylocator = mticker.MaxNLocator(3)
gl.xlabel_style = {"size": 10}
gl.ylabel_style = {"size": 10}

xtr_subplot = fig.add_subplot(gs[:5, 4:6], projection=projection)
ax3, contourf3, contour3 = plot_contour_ensemble(
    ensemble_file,
    observation=(lon, lat),
    c=0.7,
    grid_size=100,
    projection=projection,
    ax=xtr_subplot,
)
gl = ax3.gridlines(
    draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.top_labels = True
gl.bottom_labels = False
gl.left_label = False
gl.right_labels = False
gl.xformater = LongitudeFormatter(zero_direction_label=False)
gl.yformater = LatitudeFormatter(direction_label=False)
gl.xlocator = mticker.MaxNLocator(3)
gl.ylocator = mticker.MaxNLocator(3)
gl.xlabel_style = {"size": 10}
gl.ylabel_style = {"size": 10}

# Create proxies for filled contours
proxies = [
    plt.Rectangle((0, 0), 1, 1, fc=pc.get_facecolor()[0])
    for pc in contourf1.collections
]

# Combine proxies with other handles and labels
handles, labels = ax1.get_legend_handles_labels()
handles += proxies
labels += [f"{int(p * 100)}%" for p in [0.9, 0.75, 0.5, 0.25]]


fig.legend(
    handles, labels, loc="outside lower center", ncol=4, fancybox=True, shadow=True
)
plt.savefig(
    f"test_openberg/test_IK/output/tuning/{ID}/{name}.png",
    bbox_inches="tight",
)
plt.show()

contour_level = determine_contour_level(ensemble_file, (lon, lat), c=0.2)
print(f"Observation is encapsulated by contour level: {contour_level}%")
contour_level = determine_contour_level(ensemble_file, (lon, lat), c=0.4)
print(f"Observation is encapsulated by contour level: {contour_level}%")
contour_level = determine_contour_level(ensemble_file, (lon, lat), c=0.7)
print(f"Observation is encapsulated by contour level: {contour_level}%")
