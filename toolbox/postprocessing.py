from netCDF4 import Dataset, num2date
import numpy as np
import pandas as pd
import os
from toolbox.preprocessing import preprocessing
import pickle
from opendrift.models.physics_methods import skillscore_liu_weissberg
from pprint import pprint


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


def create_list_obs(input_folder, prep_params):
    list_obs = []
    for file in os.listdir(input_folder):
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


# TODO check that the match is good (the starting point should be the same)
def postprocessing(nc_file, input_folder, IDs, prep_params, ts_output: int = 3600):
    data = Dataset(nc_file)
    mod_time = num2date(data["time"][:], units=data["time"].units)
    lon = data["lon"][:].data
    lat = data["lat"][:].data
    print(f"{len(lon)} seeds")
    list_obs = create_list_obs(input_folder, prep_params)
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
            indices_mod = np.where(lon[j] < 361)[0]
            sub_lon = (lon[j, indices_mod] + 360) % 360
            sub_lat = lat[j, indices_mod]
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

    Matches = np.stack(Matches, axis=0)
    # pprint(Matches.shape)
    return Matches


def compute_SS(matches, outfile):
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
    data.to_csv(outfile, index=False)


# TODO polarplot
