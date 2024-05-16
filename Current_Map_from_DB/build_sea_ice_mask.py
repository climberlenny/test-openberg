import copernicusmarine
from pprint import pprint
import xarray as xr
import os
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from datetime import datetime

# Copernicus id and password
with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]
USERNAME = username
PASSWORD = password

ice_id = "OSISAF-GLO-SEAICE_CONC_CONT_TIMESERIES-NH-LA-OBS"

years = [2024]
month = "02"
for y in years:
    reader_ice = copernicusmarine.open_dataset(
        dataset_id=ice_id,
        username=USERNAME,
        password=PASSWORD,
        start_datetime=datetime(y, 2, 1),
        end_datetime=datetime(y, 2, 29),
    )
    mean = reader_ice.mean(dim="time")
    mean.to_netcdf(f"DATA/FOR_LENNY/Sea_ice/AVG_ice_conc_{y}_{month}.nc")
