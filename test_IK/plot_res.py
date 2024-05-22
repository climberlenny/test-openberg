from datetime import datetime, timedelta
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc
import os
from toolbox.postprocessing import plot_ensemble_mean, plot_contour_ensemble
from opendrift.models.physics_methods import skillscore_liu_weissberg


#### 7089
with open(
    "C:/Users/lenuch/ACCIBERG/test_openberg/test_IK/observations_Barents/7089.csv", "r"
) as f:
    data = pd.read_csv(f)
data["time"] = pd.to_datetime(data["time"])
selected_data = data[data["time"].dt.month < 6]

with nc.Dataset("test_openberg/test_IK/output/7089_0.nc", "r") as data:
    lon_model = data["lon"][:]
    lat_model = data["lat"][:]
    wind_speed = np.sqrt(data["x_wind"][:] ** 2 + data["y_wind"][:] ** 2)
    ocean_speed = np.sqrt(
        data["x_sea_water_velocity"][:] ** 2 + data["y_sea_water_velocity"][:] ** 2
    )
    time = data["time"]
    time = (time - time[0]) / 3600


lon = selected_data["LONGITUDE"].values
lat = selected_data["LATITUDE"].values

# plot 7089
projection = ccrs.NorthPolarStereo(34)
fig, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(12, 10))

# Add coastlines and land features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, facecolor="lightgray")

# Customize gridlines
gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.top_labels = False
gl.right_labels = False
gl.left_labels = True
gl.bottom_labels = True

# Plot the observed data
ax.plot(lon, lat, transform=ccrs.PlateCarree(), color="blue", label="Observation Track")
ax.scatter(
    lon[:: 24 * 10],
    lat[:: 24 * 10],
    transform=ccrs.PlateCarree(),
    color="blue",
    edgecolor="black",
    zorder=5,
    label="10 days positions observation",
    s=50,
)

# Plot the model data
ax.plot(
    lon_model[0],
    lat_model[0],
    transform=ccrs.PlateCarree(),
    color="orange",
    label="Model Track",
)
ax.scatter(
    lon_model[:, :: 24 * 10],
    lat_model[:, :: 24 * 10],
    transform=ccrs.PlateCarree(),
    color="orange",
    edgecolor="black",
    zorder=5,
    label="10 days positions model",
    s=50,
)

# Set the extent to focus on the area of interest
ax.set_extent([28, 40, 74, 80], crs=ccrs.PlateCarree())

# Improve the legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:3], labels[:3], loc="lower left")

# Add a title
plt.title("Observation and Model Tracks", fontsize=16)

# Show the plot
plt.show()

ensemble_file = "test_openberg/test_IK/output/7089_1_ensemble.nc"
plot_ensemble_mean(ensemble_file, projection=projection, extent=[28, 40, 74, 80])
plot_contour_ensemble(
    ensemble_file,
    observation=(lon, lat),
    c=0.2,
    grid_size=100,
    projection=projection,
    extent=[28, 40, 74, 80],
)
plot_contour_ensemble(
    ensemble_file,
    observation=(lon, lat),
    c=0.4,
    grid_size=100,
    projection=projection,
    extent=[28, 40, 74, 80],
)
plot_contour_ensemble(
    ensemble_file,
    observation=(lon, lat),
    c=0.7,
    grid_size=100,
    projection=projection,
    extent=[28, 40, 74, 80],
)


def compute_score(lon, lat, lon_mod, lat_mod):
    SS = []
    for i in range(2, len(lon)):
        ss = skillscore_liu_weissberg(lon[:i], lat[:i], lon_mod[0, :i], lat_mod[0, :i])
        SS.append(ss)

    return SS


fig, ax = plt.subplots()
score = compute_score(
    lon[: len(lon_model[0])], lat[: len(lon_model[0])], lon_model, lat_model
)
ax.plot(time[2:], score)
ax.scatter(time[2:50:3], score[:48:3], marker="+", color="k", label="score every 3h")
ax.scatter(
    time[2:50:12],
    score[:48:12],
    marker="x",
    color="r",
    zorder=10,
    s=100,
    label="score every 12h",
)
ax.set_xlabel("time(h)")
ax.set_ylabel("score")
plt.legend()
plt.show()
