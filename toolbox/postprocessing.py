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


def create_list_obs(input_folder, prep_params, files=None):
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
    return list_obs


def postprocessing(
    nc_files, input_folder, IDs, prep_params, ts_output: int = 3600, files=None
):
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
    list_obs = create_list_obs(input_folder, prep_params, files=files)
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
                previous_seed_number = new_seed_number
                new_seed_number = len(lon)
                print(f"{new_seed_number} seeds")
                indices_mod = np.where(lon[j - previous_seed_number] < 361)[0]
            sub_lon = (lon[j - previous_seed_number, indices_mod] + 360) % 360
            sub_lat = lat[j - previous_seed_number, indices_mod]
            sub_time = mod_time[indices_mod]
            test = np.where(sub_lon < 361)[0]
            # print(sub_lon)
            # pprint(sub_time)
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


def statistics(data: pd.DataFrame, by=["wind model", "ocean model"], outfolder=None):
    if not os.path.exists(outfolder):
        os.mkdir(os.path.join(".", outfolder))
    median = data.loc[:, ["SS"] + by].groupby(by=by).median()
    mean = data.loc[:, ["SS"] + by].groupby(by=by).mean()
    std = data.loc[:, ["SS"] + by].groupby(by=by).std()
    min = data.loc[:, ["SS"] + by].groupby(by=by).min()
    max = data.loc[:, ["SS"] + by].groupby(by=by).max()
    combined_df = pd.concat(
        [median, mean, std, min, max],
        keys=["median", "mean", "std", "min", "max"],
        axis=1,
    )
    combined_df.to_csv(os.path.join(outfolder, "statistics.txt"), sep="\t")

    ax = data.boxplot(column="SS", by=by, figsize=(10, 6))

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
    x = r * np.cos(np.radians(theta))
    y = r * np.sin(np.radians(theta))
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
    THETA = np.array(THETA)
    outsiders = (R > 5).sum()
    # Convert polar coordinates to Cartesian coordinates
    X, Y = polar_to_cartesian(THETA, R)

    # Create a 2D histogram (hexbin) to bin the data
    bins = 1000

    # Plot the density
    fig, ax = plt.subplots()
    hb = ax.hexbin(X, Y, cmap="viridis", gridsize=1000, bins="log")
    plt.colorbar(hb, label="Density")
    circle = plt.Circle(
        (0, 0), radius=1, color="none", linestyle="dashed", edgecolor="black", alpha=0.5
    )
    ax.add_artist(circle)
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    plt.title(f"Polar Plot \n {outsiders} points out of plot domain")
    plt.gca().set_aspect(
        "equal", adjustable="box"
    )  # Set aspect ratio to make it look polar
    plt.savefig(outfile, bbox_inches="tight")
    plt.show()
