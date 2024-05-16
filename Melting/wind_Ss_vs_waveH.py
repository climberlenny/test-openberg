from datetime import datetime, timedelta
from opendrift.models.openberg import OpenBerg
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc
import copernicusmarine
from pprint import pprint
from toolbox.preprocessing import Nearest2DInterpolator
from scipy.signal import correlate2d
from scipy.stats import pearsonr


# wind
wind_data = nc.Dataset("DATA/FOR_LENNY/WIND_MODELS/2021/ERA5/6h.10m_wind_2021.nc")

lon = wind_data["longitude"]
lat = wind_data["latitude"]
x_wind = wind_data["x_wind"][0]
y_wind = wind_data["y_wind"][0]
lon, lat = np.meshgrid(lon, lat)
V_wind = np.sqrt(x_wind**2 + y_wind**2)
Ss = -5 + np.sqrt(32 + V_wind)

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND)
gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.right_labels = False
Vmap = ax.pcolormesh(lon, lat, Ss, transform=ccrs.PlateCarree())
plt.colorbar(Vmap)
ax.set_extent([-60, -20, 50, 80], ccrs.PlateCarree())
plt.show()

with open("../ACCIBERG/LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]

USERNAME = username
PASSWORD = password

wave = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"
reader_wave = copernicusmarine.open_dataset(
    dataset_id=wave,
    username=USERNAME,
    password=PASSWORD,
    start_datetime=datetime(2021, 1, 1, 0),
    end_datetime=datetime(2021, 1, 1, 6),
    minimum_latitude=50,
    maximum_latitude=70,
    minimum_longitude=-60,
    maximum_longitude=-20,
)

H_wave = reader_wave["VHM0"].mean(dim="time")
lon_wave = reader_wave["longitude"][:]
lat_wave = reader_wave["latitude"][:]
lon_wave, lat_wave = np.meshgrid(lon_wave, lat_wave)


fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND)
gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.right_labels = False
wave = ax.pcolormesh(lon_wave, lat_wave, H_wave, transform=ccrs.PlateCarree())
plt.colorbar(wave)
ax.set_extent([-60, 0, 50, 90], ccrs.PlateCarree())
plt.show()


interp = Nearest2DInterpolator(lon, lat, lon_wave, lat_wave)
interpolated_wind = interp(Ss)


fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo()})
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND)
gl = ax.gridlines(
    draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="dotted"
)
gl.right_labels = False
wave = ax.pcolormesh(
    lon_wave,
    lat_wave,
    H_wave - interpolated_wind,
    transform=ccrs.PlateCarree(),
    cmap="bwr",
    vmin=-3,
    vmax=3,
)
plt.colorbar(wave)
ax.set_extent([-60, 0, 50, 90], ccrs.PlateCarree())
plt.show()
# Pearson = np.corrcoef(H_wave.data.flatten(), interpolated_wind.data.flatten())


# # Define the patch size
# patch_size = (3, 3)

# # Calculate the number of patches in each dimension
# num_patches_y = H_wave.shape[0] // patch_size[0]
# num_patches_x = H_wave.shape[1] // patch_size[1]

# # Initialize an array to store the Pearson correlation coefficients
# correlation_map = np.zeros((num_patches_y, num_patches_x))

# # Iterate through the patches and compute the Pearson correlation coefficient
# for i in range(num_patches_y):
#     for j in range(num_patches_x):
#         # Define the boundaries of the current patch
#         y_start = i * patch_size[0]
#         y_end = y_start + patch_size[0]
#         x_start = j * patch_size[1]
#         x_end = x_start + patch_size[1]

#         # Extract the patches from the arrays
#         patch1 = H_wave[y_start:y_end, x_start:x_end].data
#         patch2 = interpolated_wind[y_start:y_end, x_start:x_end].data

#         # Compute the Pearson correlation coefficient for the current patch
#         correlation_coefficient, _ = pearsonr(patch1.flatten(), patch2.flatten())

#         # Store the correlation coefficient in the map
#         correlation_map[i, j] = correlation_coefficient

# # Plotting the correlation coefficient map
# fig, ax = plt.subplots(subplot_kw={"projection": ccrs.NorthPolarStereo()})
# ax.add_feature(cfeature.COASTLINE)
# ax.add_feature(cfeature.BORDERS)
# ax.add_feature(cfeature.LAND)

# # Define the extent of the map (adjust as needed)
# ax.set_extent([-40, -20, 50, 60], ccrs.PlateCarree())

# # Plot the correlation map
# im = ax.imshow(
#     correlation_map,
#     cmap="coolwarm",
#     extent=[-40, -20, 50, 60],
#     origin="lower",
#     transform=ccrs.PlateCarree(),
#     interpolation="nearest",
# )

# # Add a colorbar
# cbar = plt.colorbar(
#     im, ax=ax, orientation="horizontal", label="Pearson Correlation Coefficient"
# )

# # Show the plot
# plt.show()
