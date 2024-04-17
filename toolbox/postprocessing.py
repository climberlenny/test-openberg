from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import os
from toolbox.preprocessing import preprocessing
import pickle
from opendrift.models.physics_methods import skillscore_liu_weissberg
from pprint import pprint
import matplotlib.pyplot as plt
import pyproj
from matplotlib.colors import ListedColormap, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from scipy.stats import circmean


def compute_IDs(files: list, input_folder: str, outfile: str, prep_params: dict):
    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            IDs = pickle.load(f)
        return IDs
    else:
        IDs = []
        i = 0
        for file in files[:]:
            fp = os.path.join(input_folder, file)
            with open(fp) as f:
                data = preprocessing(
                    fp,
                    column_names=prep_params["column_names"],
                    date_format=prep_params["date_format"],
                    time_thresh=prep_params["time_thresh"],
                    dist_thresh=prep_params["dist_thresh"],
                    timestep_interpolation=prep_params["ts_interp"],
                    No_column=prep_params["No_column"],
                )
                for df in data:
                    IDs.append(
                        i + len(df.index[:: int(86400 / prep_params["ts_interp"])])
                    )
                    i += len(df.index[:: int(86400 / prep_params["ts_interp"])])
        with open(outfile, "wb") as f:
            pickle.dump(IDs, f)
        return IDs


def match_ID_obs(line: int, IDs: list, files: list):
    wh = np.where(line < np.array(IDs))[0][0]
    # print(wh)
    previous_ID = IDs[wh - 1] if wh > 0 else wh
    real_ID = line - previous_ID
    # print(real_ID)
    return files[wh], real_ID


def create_list_obs(input_folder, prep_params, files=None, outfile=None):
    if os.path.exists(outfile):
        with open(outfile, "rb") as f:
            list_obs = pickle.load(f)
        return list_obs
    list_obs = []
    if files is None:
        files = os.listdir(input_folder)
    for file in files:
        # print(f"filename : {file}")
        with open(os.path.join(input_folder, file)) as f:
            obs = preprocessing(
                f,
                column_names=prep_params["column_names"],
                date_format=prep_params["date_format"],
                time_thresh=prep_params["time_thresh"],
                dist_thresh=prep_params["dist_thresh"],
                timestep_interpolation=prep_params["ts_interp"],
                No_column=prep_params["No_column"],
            )
        for i in range(len(obs)):
            list_obs.append(obs[i])

    with open(outfile, "wb") as f:
        pickle.dump(list_obs, f)
    return list_obs


def postprocessing(
    nc_files,
    input_folder,
    IDs,
    prep_params,
    ts_output: int = 3600,
    files=None,
    outfile=None,
):
    if not files is None:
        group = list(range(0, len(files), 100))
        group.append(None)
    f = 0
    data = Dataset(nc_files[f])
    mod_time = num2date(data["time"][:], units=data["time"].units)
    lon = data["lon"][:].data
    lat = data["lat"][:].data
    previous_seed_number = 0
    new_seed_number = len(lon)
    print(f"{new_seed_number} seeds")
    list_obs = create_list_obs(input_folder, prep_params, files=files, outfile=outfile)
    print(len(list_obs))
    # assert len(list_obs) == len(IDs)
    Matches = []
    for i, sub_obs in enumerate(list_obs):
        i_lim0 = 0
        if i > 0:
            i_lim0 = IDs[i - 1]
        i_lim = IDs[i]
        # print(i_lim)
        lim_ID = len(sub_obs)
        # print(lim_ID)
        for j in range(i_lim0, i_lim):
            _, RID = match_ID_obs(j, IDs, list_obs)
            print(f"seed {j} - RID {RID}")
            start_ID_obs = RID * int(86400 / prep_params["ts_interp"])
            end_ID_obs = (
                start_ID_obs + (int(86400 / prep_params["ts_interp"]) + 1)
                if (start_ID_obs + (int(86400 / prep_params["ts_interp"]) + 1) < lim_ID)
                else None
            )

            sub_df = sub_obs.iloc[
                start_ID_obs : end_ID_obs : int(ts_output / prep_params["ts_interp"])
            ]
            # pprint(sub_df)
            if len(sub_df) < int(86400 / ts_output + 1):
                print("Warning : obs too short !")
                continue
            if j - previous_seed_number < new_seed_number:
                indices_mod = np.where(lon[j - previous_seed_number] < 361)[0]
            else:
                f += 1
                data = Dataset(nc_files[f])
                mod_time = num2date(data["time"][:], units=data["time"].units)
                lon = data["lon"][:].data
                lat = data["lat"][:].data
                previous_seed_number += new_seed_number
                new_seed_number = len(lon)
                print(f"{new_seed_number} seeds")
                indices_mod = np.where(lon[j - previous_seed_number] < 361)[0]
            sub_lon = (lon[j - previous_seed_number, indices_mod] + 360) % 360
            sub_lat = lat[j - previous_seed_number, indices_mod]
            sub_time = mod_time[indices_mod]
            test = np.where(sub_lon < 361)[0]
            # print(sub_lon)
            # pprint(sub_time)
            if np.where(np.logical_and(sub_lat > 66, sub_lat < 69))[0]:
                print("DEBUG : ", sub_lat)
            if len(test) < int(86400 / ts_output + 1):
                print("Warning : simulation too short !")
                pass
            else:
                print("It's a Match !")
                # reshape lon lat
                sub_lon = np.reshape(sub_lon, (len(sub_lon), 1))
                sub_lat = np.reshape(sub_lat, (len(sub_lon), 1))
                obs_lon = sub_df.loc[:, "Longitude"].values.reshape((len(sub_lon), 1))
                obs_lat = sub_df.loc[:, "Latitude"].values.reshape((len(sub_lon), 1))
                match_obs = np.hstack((obs_lon, obs_lat))
                match_mod = np.hstack((sub_lon, sub_lat))
                _match = np.stack((match_obs, match_mod), axis=0)
                Matches.append(_match)

    if len(Matches) > 0:
        Matches = np.stack(Matches, axis=0)
    # pprint(Matches.shape)
    return Matches


def compute_SS(matches, outfile, save=False):
    LON = []
    LAT = []
    SS = []
    # plt.hist(matches[:, 0, :, 1].flatten(), bins=100)
    # plt.show()
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]
        ss = skillscore_liu_weissberg(lon_obs, lat_obs, lon_mod, lat_mod)
        lon = obs[0, 0]
        lat = obs[0, 1]
        LON.append(lon)
        LAT.append(lat)
        SS.append(ss)

    data = {"LON": LON, "LAT": LAT, "SS": SS}
    data = pd.DataFrame(data)
    if save:
        data.to_csv(outfile, index=False)
    return data


def statistics(
    data: pd.DataFrame, by=["wind model", "ocean model"], outfolder=None, **kwargs
):
    if not os.path.exists(outfolder):
        os.mkdir(os.path.join(".", outfolder))
    if not by is None:
        median = data.loc[:, ["SS"] + by].groupby(by=by).median()
        mean = data.loc[:, ["SS"] + by].groupby(by=by).mean()
        std = data.loc[:, ["SS"] + by].groupby(by=by).std()
        _min = data.loc[:, ["SS"] + by].groupby(by=by).min()
        _max = data.loc[:, ["SS"] + by].groupby(by=by).max()
        combined_df = pd.concat(
            [median, mean, std, _min, _max],
            keys=["median", "mean", "std", "min", "max"],
            axis=1,
        )
        combined_df.to_csv(os.path.join(outfolder, "statistics.txt"), sep="\t")

        ax = data.boxplot(column="SS", by=by, figsize=(10, 6), **kwargs)

        # Adjust font size
        plt.xticks(fontsize=8)

        # Rotate and align the x-axis labels vertically
        xticklabels = [
            label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(xticklabels)
        plt.savefig(os.path.join(outfolder, "boxplot.png"))
    else:
        median = data.loc[:, ["SS"]].median()
        mean = data.loc[:, ["SS"]].mean()
        std = data.loc[:, ["SS"]].std()
        _min = data.loc[:, ["SS"]].min()
        _max = data.loc[:, ["SS"]].max()
        stats = {"median": median, "mean": mean, "std": std, "min": _min, "max": _max}
        stats = pd.DataFrame(stats)
        stats.to_csv(os.path.join(outfolder, "statistics.txt"), sep="\t")
        ax = data.boxplot(column="SS", figsize=(10, 6))

        # Adjust font size
        plt.xticks(fontsize=8)

        # Rotate and align the x-axis labels vertically
        xticklabels = [
            label.get_text().replace(" ", "\n") for label in ax.get_xticklabels()
        ]
        ax.set_xticklabels(xticklabels)
        plt.savefig(os.path.join(outfolder, "boxplot.png"))


# TODO density plot in polar coordinate
def polarplot(matches, outfile, SS=None):
    THETA = []
    R = []
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]

        geod = pyproj.Geod(ellps="WGS84")

        # observation = ref
        azimuth_alpha, a2, distance0 = geod.inv(
            lon_obs[0], lat_obs[0], lon_obs[-1], lat_obs[-1]
        )
        vel0 = distance0 / 86400
        # simulation
        azimuth_betha, a2, distance1 = geod.inv(
            lon_mod[0], lat_mod[0], lon_mod[-1], lat_mod[-1]
        )
        vel1 = distance1 / 86400

        # calculate the coordinates in the polar plot
        theta = (azimuth_betha - azimuth_alpha + 360) % 360
        r = vel1 / vel0 if vel0 > 0 else 100
        THETA.append(theta)
        R.append(r)

    colors = [
        "#F8831C",  # orange
        "#1C26F8",  # blue
        "#000000",  # black
    ]

    R = np.array(R)
    THETA = np.array(THETA)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N", offset=0)
    ax.set_theta_direction(-1)
    ax.grid(True)
    # Set radial ticks
    radial_ticks = [0.2, 0.4, 0.6, 0.8, 1, 2]
    # Set radial tick labels, keep empty strings for values less than 1
    radial_ticklabels = ["" if tick < 1 else str(tick) for tick in radial_ticks]
    ax.set_rticks(radial_ticks)
    ax.set_yticklabels(radial_ticklabels)

    if SS is None:

        mask1 = np.where(np.logical_or(R < 0.5, np.logical_and(R > 1.5, R <= 2)))[0]
        mask2 = np.where(np.logical_and(R >= 0.5, R <= 1.5))[0]
        mask3 = np.where(R > 2)[0]
        # orange
        ax.scatter(
            THETA[mask1],
            R[mask1],
            zorder=10,
            color=colors[0],
            label="R < 0.5 or R in [1.5,2] ",
        )
        # blue
        ax.scatter(
            THETA[mask2], R[mask2], zorder=10, color=colors[1], label="R in [0.5,1.5]"
        )
        # black
        ax.scatter(
            THETA[mask3],
            R[mask3] / R[mask3] + 1,
            zorder=10,
            color=colors[2],
            label="R > 2",
        )

        plt.legend(loc="center left", bbox_to_anchor=(1, 0.1))
    else:
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=1)  # Assuming SS values range from 0 to 1
        mask = np.where(R <= 2)[0]
        colors = cmap(norm(SS[mask]))
        ax.scatter(
            THETA[mask],
            R[mask],
            zorder=10,
            color=colors,
        )
        mask = np.where(R > 2)[0]
        colors = cmap(norm(SS[mask]))
        ax.scatter(
            THETA[mask],
            R[mask] / R[mask] + 1,
            zorder=10,
            color=colors,
        )
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label("SS Value")
    ax.set_rmin(0)
    ax.set_rmax(2.1)
    ax.set_title(f"{len(R)} points")
    plt.savefig(outfile, bbox_inches="tight")
    # plt.show()


def polar_to_cartesian(theta, r):
    """Convert polar coordinates to Cartesian coordinates."""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def polarplot2(matches, outfile, SS=None):
    THETA = []
    R = []
    for match in matches:
        obs = match[0]
        mod = match[1]
        lon_obs = obs[:, 0]
        lat_obs = obs[:, 1]
        lon_mod = mod[:, 0]
        lat_mod = mod[:, 1]

        geod = pyproj.Geod(ellps="WGS84")

        # observation = ref
        azimuth_alpha, a2, distance0 = geod.inv(
            lon_obs[0], lat_obs[0], lon_obs[-1], lat_obs[-1]
        )
        vel0 = distance0 / 86400
        # simulation
        azimuth_betha, a2, distance1 = geod.inv(
            lon_mod[0], lat_mod[0], lon_mod[-1], lat_mod[-1]
        )
        vel1 = distance1 / 86400

        # calculate the coordinates in the polar plot
        theta = (azimuth_betha - azimuth_alpha + 360) % 360
        r = vel1 / vel0 if vel0 > 0 else 10
        THETA.append(theta)
        R.append(r)

    R = np.array(R)
    r_avg_with_outsiders = np.mean(R)
    THETA = np.array(THETA)
    theta_avg_with_outsiders = circmean(THETA)
    print(theta_avg_with_outsiders, r_avg_with_outsiders, np.max(R))
    # print(len(SS),len(R))
    outsiders = (R > 2).sum()
    # Convert polar coordinates to Cartesian coordinates
    X, Y = polar_to_cartesian(np.pi / 2 - np.radians(THETA[R <= 2]), R[R <= 2])
    x_outsiders, y_outsiders = polar_to_cartesian(
        np.pi / 2 - np.radians(theta_avg_with_outsiders), r_avg_with_outsiders
    )
    r_avg_insiders = np.mean(R[R <= 2])
    theta_avg_insiders = circmean(THETA[R <= 2])
    x_insiders, y_insiders = polar_to_cartesian(
        np.pi / 2 - np.radians(theta_avg_insiders), r_avg_insiders
    )

    # Plot the density
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(X, Y, cmap="viridis", gridsize=50, bins="log")
    plt.colorbar(hb, label="Density")
    ax.scatter(0, 1, marker="x", s=200, c="r", label="target")
    ax.scatter(
        x_insiders,
        y_insiders,
        marker="o",
        s=50,
        c="r",
        label=f"mean without outsiders ({np.round(r_avg_insiders,1)},{np.round(theta_avg_insiders,1)})",
    )
    ax.plot(
        np.linspace(-2, 2, 100),
        np.tan(np.pi / 2 - np.deg2rad(theta_avg_insiders)) * np.linspace(-2, 2, 100),
        color="k",
        linestyle="--",
        label=f"average deviation : {np.round(theta_avg_insiders)}",
    )
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    plt.title(f"Polar Plot \n {outsiders} points out of plot domain")
    plt.legend(loc="lower left")
    plt.gca().set_aspect(
        "equal", adjustable="box"
    )  # Set aspect ratio to make it look polar
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()


def plot_current_map(data, outfolder, model):
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

    data.loc[:, "LON"] = (data.loc[:, "LON"] + 360) % 360
    data = data.sort_values(by="LON")
    lon = data.loc[:, "LON"].values
    lat = data.loc[:, "LAT"].values
    SkillScore = data.loc[:, "SS"].values

    # plt.hist(lat,bins=100)
    # plt.show()

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
        lat_max = min(90, lat_chunk.max() + 1)
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
        os.path.join(outfolder, f"{model}_SS_map_global.png"),
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.show()
