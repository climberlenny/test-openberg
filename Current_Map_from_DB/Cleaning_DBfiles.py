import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pyproj
import shutil
import pandas as pd
import cftime
from toolbox.preprocessing import Nearest2DInterpolator
from tqdm import tqdm
from scipy.stats import circmean
from pprint import pprint

dirname = "DATA/FOR_LENNY/DRIFTERS"
dir2 = "DATA/FOR_LENNY/removed_drifters"
dir_csv = "DATA/FOR_LENNY/DB_csv2"

files = os.listdir(dirname)


def access_ice_mask(ice_folder, year, month):
    filename = f"AVG_ice_conc_{year}_{str(month).zfill(2)}.nc"
    file_path = os.path.join(ice_folder, filename)
    if os.path.isfile(file_path):
        return file_path
    else:
        return f"File '{filename}' not found in directory '{ice_folder}'."


def check_ice_buoys(mask, lon, lat):
    if lat > 80:
        pass
    # interp = Nearest2DInterpolator(mask[0], mask[1], np.array([lon]), np.array([lat]))
    index = np.abs(mask[0] - lon) + np.abs(mask[1] - lat)
    jj, ii = np.unravel_index(
        index.argmin(), index.shape
    )  # interpolation nearest neighbor
    ice_conc = mask[2][jj, ii]
    if np.isnan(ice_conc) or ice_conc < 0:
        return False
    elif ice_conc >= 0.15:
        return True
    else:
        return False


def preparing_ice_mask(filename, year):
    reader = nc.Dataset(filename)

    if year >= 2021:
        lon = reader["longitude"][:].data
        lat = reader["latitude"][:].data
        lon, lat = np.meshgrid(lon, lat)
        ice_conc = reader["ice_conc"][:].data
    else:
        lon = reader["lon"][:].data
        lat = reader["lat"][:].data
        ice_conc = reader["ice_conc"][0].data

    return (lon, lat, ice_conc)


def circular_mean(x):
    return round(np.rad2deg(circmean(np.deg2rad(x))), 2)


stats = {
    "name": {},
    "src_ptf_cat": {},
    "Nvalid": 0,
    "tot": 0,
}
for file in tqdm(files):

    data = nc.Dataset(os.path.join(dirname, file))
    ptf_code = data.platform_code
    src_ptf_cat = data.source_platform_category_code
    ptf_name = data.platform_name

    lat_min = float(data.geospatial_lat_min)
    lat_max = float(data.geospatial_lat_max)
    lon_min = float(data.geospatial_lon_min)
    lon_max = float(data.geospatial_lon_max)

    if os.path.exists(os.path.join(dir_csv, f"{ptf_code}_DB.csv")):
        continue

    start = datetime.fromisoformat(data.time_coverage_start)
    end = datetime.fromisoformat(data.time_coverage_end)

    # print(src_ptf_cat, ptf_name)
    stats["tot"] += 1
    if lat_min < 50:
        print("trajectory out of the arctic area !")
        data.close()
        shutil.move(os.path.join(dirname, file), os.path.join(dir2, file))
        continue
    if (end - start).days < 2:
        print("trajectory too short in time !")
        data.close()
        shutil.move(os.path.join(dirname, file), os.path.join(dir2, file))
        continue
    if ptf_name != "Drifting Buoy":
        print("Incorrect name !")
        data.close()
        shutil.move(os.path.join(dirname, file), os.path.join(dir2, file))
        continue
    # if not src_ptf_cat in [42, 49]:
    #     print("code ID incorrect !")
    #     continue

    geod = pyproj.Geod(ellps="WGS84")
    azimuth_forward, a2, distance = geod.inv(lon_min, lat_min, lon_max, lat_max)
    if distance / 1000 < 50:
        print("trajectory too short in space !")
        data.close()
        shutil.move(os.path.join(dirname, file), os.path.join(dir2, file))
        continue

    time_qc = data["TIME_QC"][:].data
    time_mask = time_qc == 1
    position_qc = data["POSITION_QC"][:].data
    pos_mask = position_qc == 1
    mask = np.logical_and(time_mask, pos_mask)

    lon = (data["LONGITUDE"][mask].data + 360) % 360
    lat = data["LATITUDE"][mask].data
    time = pd.DatetimeIndex(
        nc.num2date(
            data["TIME"][mask],
            units=data["TIME"].units,
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
    )
    df = pd.DataFrame(
        {"lon": lon, "lat": lat, "month": time.month, "year": time.year}, index=time
    )
    df_per_month = df.groupby(["month", "year"])[["lon", "lat"]].agg(circular_mean)
    df_per_month.loc[:, "lon"] = df_per_month.loc[:, "lon"] % 360
    df_per_month.loc[:, "lon"] = (
        (df_per_month.loc[:, "lon"] + 180) % 360
    ) - 180  # convert into range (-180,180)
    # pprint(df_per_month)
    # stop
    for my in df_per_month.index:
        c = False
        month, year = my
        ice_file = access_ice_mask("DATA/FOR_LENNY/Sea_ice", year, month)
        mask_ice = preparing_ice_mask(ice_file, year)
        isinice = check_ice_buoys(
            mask_ice, df_per_month.loc[my, "lon"], df_per_month.loc[my, "lat"]
        )
        if isinice:
            print("trajectory in Ice !")
            data.close()
            shutil.move(os.path.join(dirname, file), os.path.join(dir2, file))
            c = True
            break
    if c:
        continue
    if len(lon) < 2:
        print("trajectory too short in space !")
        data.close()
        shutil.move(os.path.join(dirname, file), os.path.join(dir2, file))
        continue

    if ptf_name in stats["name"].keys():
        stats["name"][ptf_name] += 1
    else:
        stats["name"][ptf_name] = 1
    if src_ptf_cat in stats["src_ptf_cat"].keys():
        stats["src_ptf_cat"][src_ptf_cat] += 1
    else:
        stats["src_ptf_cat"][src_ptf_cat] = 1

    stats["Nvalid"] += 1
    # save trajectory in a csv file
    time = data["TIME"][mask].data
    time = pd.to_datetime(
        time,
        unit="D",
        origin=datetime.strptime("1950-01-01T00:00:00Z", "%Y-%m-%dT%H:%M:%SZ"),
    )
    time = time.strftime("%Y-%m-%d %H:%M:%S")

    df = {
        "Latitude": lat,
        "Longitude": lon,
        "date": time,
    }
    if not os.path.exists(os.path.join(dir_csv, f"{ptf_code}_DB.csv")):
        df = pd.DataFrame(data=df)
        df.to_csv(os.path.join(dir_csv, f"{ptf_code}_DB.csv"), index=False)
    data.close()


def cut_data(df: pd.DataFrame, threshold: int = 7) -> list[pd.DataFrame]:
    dfs = []

    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values(by=["date"])
    # Find indices where timedelta exceeds N days
    indices = df.index[df["date"].diff() > pd.Timedelta(days=threshold)]

    # Split DataFrame based on jumps
    dfs = []
    start_idx = 0
    for end_idx in indices:
        dfs.append(df.iloc[start_idx:end_idx])
        start_idx = end_idx
    dfs.append(df.iloc[start_idx:])
    return dfs


dir_csv = "DATA/FOR_LENNY/DB_csv2"

files = os.listdir(dir_csv)
for file in files:
    path = os.path.join(dir_csv, file)
    data = pd.read_csv(path)

    dfs = cut_data(data)

    if len(dfs) > 1:
        os.remove(path)
        print()
        print(f"cutting {file}")
        print()
        for i, sub_df in enumerate(dfs):
            start = sub_df.loc[sub_df.index[0], "date"]
            end = sub_df.loc[sub_df.index[-1], "date"]
            duration = (end - start).days
            print(f"DataFrame {i+1} : {duration} days")
            # print(sub_df)
            if duration > 2:
                if not os.path.exists(
                    os.path.join(dir_csv, file.replace(".csv", f"_{i+1}.csv"))
                ):
                    sub_df.to_csv(
                        os.path.join(dir_csv, file.replace(".csv", f"_{i+1}.csv")),
                        index=False,
                    )
                print(f"save sub df {i+1}")
            else:
                print(f"delete sub df {i+1}")
