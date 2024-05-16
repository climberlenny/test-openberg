import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc
import os
import pickle


input_folder = "test_openberg/test_wave_models/output"

stokes = ["SD", "noSD"]
wave = ["Wave", "noWave"]
color_list = ["#a6611a", "#018571", "#d01c8b", "r"]
linestyle_list = ["-", "-.", ":", "dashdot"]

data = pd.read_csv(
    "test_openberg/test_wave_models/output/SSglobal_noSD_noWave_no_acc.csv"
)
best_index = data.sort_values("SS", ascending=False)[:5]
print(data.sort_values("SS", ascending=False)[:5])
with open("test_openberg/test_wave_models/output/matches.pickle", "rb") as f:
    Matches = pickle.load(f)

for rank, i in enumerate(best_index.index):
    k = 0
    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
    )
    gl.right_labels = False
    for sd in stokes:
        for w in wave:
            nc_data = nc.Dataset(os.path.join(input_folder, f"global_{sd}_{w}_acc.nc"))

            lon = nc_data["lon"][i + 1].data
            lat = nc_data["lat"][i + 1].data
            lon[lon < 361] = (lon[lon < 361] + 360) % 360
            print(i + 1, lon[lon < 361][0], lat[lon < 361][0])
            ax.plot(
                lon,
                lat,
                transform=ccrs.PlateCarree(),
                color=color_list[k],
                linestyle=linestyle_list[k],
                label=f"{sd}_{w}",
            )
            k += 1
    nc_data = nc.Dataset(os.path.join(input_folder, f"global_noSD_noWave_no_acc.nc"))
    lon = nc_data["lon"][i + 1].data
    lat = nc_data["lat"][i + 1].data
    lon[lon < 361] = (lon[lon < 361] + 360) % 360
    ax.plot(
        lon,
        lat,
        transform=ccrs.PlateCarree(),
        color="b",
        linestyle="--",
        label="No acc",
    )
    obs = Matches[i, 0]
    print(obs)
    if i == 250:
        obs = Matches[14, 0]
        ax.plot(
            obs[:, 0],
            obs[:, 1],
            transform=ccrs.PlateCarree(),
            color="k",
            label=f"observation {14}",
        )
    else:
        ax.plot(
            obs[:, 0],
            obs[:, 1],
            transform=ccrs.PlateCarree(),
            color="k",
            label=f"observation {i}",
        )

    ax.set_extent(
        [
            lon[lon < 361].min() - 0.25,
            lon[lon < 361].max() + 0.25,
            lat[lon < 361].min() - 0.25,
            lat[lon < 361].max() + 0.25,
        ]
    )
    ax.set_title(f"Wave model Benchmark top {rank+1}")
    plt.legend()
    plt.show()
