import os
from toolbox.postprocessing import postprocessing, compute_IDs, compute_SS
from pprint import pprint
import pandas as pd
from matplotlib.colors import ListedColormap
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np


output_folder = "test_openberg/Current_Map_from_DB/output"
input_folder = "DATA/FOR_LENNY/DB_csv2"

dffiles = pd.read_csv("test_openberg/Current_Map_from_DB/sorted_files.csv")
files = dffiles.loc[:, "files"].values

PATH = "test_openberg/Current_Map_from_DB"

columns = {"Latitude": "Latitude", "Longitude": "Longitude", "date": "time"}

prep_params = {
    "column_names": columns,
    "date_format": "%Y-%m-%d %H:%M:%S",
    "time_thresh": 7,
    "dist_thresh": 0,
    "ts_interp": 3600,
    "No_column": False,
}


IDs = compute_IDs(
    files,
    input_folder=input_folder,
    outfile=os.path.join(PATH, "list_ID.pickle"),
    prep_params=prep_params,
)
print(IDs)


def plot_map(model):
    file = f"test_openberg/Current_Map_from_DB/output/SS_{model}.csv"

    colors = [
        "#f7f7f7",
        "#542788",
        "#998ec3",
        "#d8daeb",
        "#fee0b6",
        "#f1a340",
        "#b35806",
    ]
    custom_cmap = ListedColormap(colors)

    data = pd.read_csv(file)
    data.loc[:, "Longitude"] = (data.loc[:, "Longitude"] + 360) % 360
    data = data.sort_values(by="Longitude")
    lon = data.loc[:, "Longitude"].values
    lat = data.loc[:, "Latitude"].values
    SkillScore = data.loc[:, "SkillScore"].values

    fig, ax = plt.subplots(
        subplot_kw={"projection": ccrs.NorthPolarStereo()},
        figsize=(8, 8),
        dpi=200,
    )
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND)
    gl = ax.gridlines(
        draw_labels=False, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
    )
    gl.right_labels = False

    # Define the number of chunks
    num_chunks = 100
    # Define radius
    radius = 0.5

    # Calculate the number of lon points in each chunk
    chunk_size = len(lon) // num_chunks
    # Iterate over chunks
    for i in range(num_chunks):
        # Calculate the start and end indices for the chunk
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else len(lon)

        # Extract lon and lat values for the chunk
        lon_chunk = lon[start_idx:end_idx]
        lmin = lon_chunk.min()
        lmax = lon_chunk.max()
        # if lmin < l1 - 20 or lmax > l2 + 20:
        #     continue
        lat_chunk = lat[start_idx:end_idx]
        ss_chunk = SkillScore[start_idx:end_idx]

        # Calcul des limites
        lon_min = max(0, lon_chunk.min() - 1)
        lon_max = min(360, lon_chunk.max() + 1)
        lat_min = max(-90, lat_chunk.min() - 1)
        lat_max = min(80, lat_chunk.max() + 1)
        grid_resolution = 0.1

        # Create lon-lat grid for the chunk
        lon_grid, lat_grid = np.meshgrid(
            np.arange(lon_min, lon_max, grid_resolution),
            np.arange(lat_min, lat_max, grid_resolution),
        )
        # Calcul des distances à tous les points de données pour le chunk
        distances_chunk = np.sqrt(
            (lon_chunk[:, None, None] - lon_grid) ** 2
            + (lat_chunk[:, None, None] - lat_grid) ** 2
        )
        # Initialize third grid
        third_grid = np.full(lon_grid.shape, np.nan)

        # Vérifier si un point de données est dans le rayon pour chaque point de la grille
        within_radius = distances_chunk <= radius
        mask_isData = np.any(within_radius, axis=0)
        # print(np.where(mask_isData))

        average_SS = np.nansum(
            np.reshape(ss_chunk, (len(ss_chunk), 1, 1)) * within_radius, axis=0
        ) / np.nansum(within_radius, axis=0)
        # print(np.nanmin(average_SS), np.nanmax(average_SS), np.nanmean(average_SS))

        thresholds = [0, 0.005, 0.1, 0.2, 0.3, 0.5, 1]

        # Assigner les valeurs en fonction des seuils
        for i, (threshold1, threshold2) in enumerate(
            zip(thresholds[:-1], thresholds[1:])
        ):
            if i < 3:
                inThreshold = np.logical_and(
                    average_SS < threshold2, threshold1 < average_SS
                )
                third_grid[mask_isData & inThreshold] = i + 1
            else:
                inThreshold = np.logical_and(
                    average_SS < threshold2, threshold1 < average_SS
                )
                third_grid[mask_isData & inThreshold] = i + 1

        mp = plt.pcolormesh(
            lon_grid,
            lat_grid,
            third_grid,
            cmap=custom_cmap,
            transform=ccrs.PlateCarree(),
            vmin=0,
            vmax=6,
        )

    plt.colorbar(
        mappable=mp,
        ax=ax,
        ticks=[0, 1, 2, 3, 4, 5, 6],
        format=mticker.FixedFormatter(
            [
                "No data",
                "SS < 0.005",
                "0.005 <= SS < 0.1",
                "0.1 <= SS < 0.2",
                "0.2 <= SS < 0.3",
                "0.3 <= SS < 0.5",
                "0.5 <= SS <= 1",
            ]
        ),
    )
    ax.set_title(f"{model} SkillScore Map")
    # if l1 == 180:
    #     l1 = 181
    ax.set_extent([0, 360, 60, 90], ccrs.PlateCarree())
    plt.savefig(
        os.path.join(output_folder, f"SS_map_global_{model}.png"),
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.show()


group = list(range(0, len(files), 100))
group.append(None)

models = ["GLOB"]
for model in models:
    SS = []
    nc_f = [
        os.path.join(output_folder, f"global_{model}_{i}_run.nc") for i in group[0:-1]
    ]
    Matches = postprocessing(
        nc_f,
        input_folder,
        IDs,
        prep_params,
        ts_output=43200,
        files=files,
    )
    if len(Matches) > 0:
        SS_data = compute_SS(
            Matches,
            outfile=os.path.join(output_folder, f"SS_{model}.csv"),
            save=False,
        )
        SS.append(SS_data)
        # polarplot2(
        #     Matches,
        #     os.path.join(output_folder, f"polarplot.png"),
        # data_SS.loc[:, "SS"],
        # )
    # pprint(Matches)
    SS = pd.concat(SS)
    SS.to_csv(os.path.join(output_folder, f"SS_{model}.csv"))
