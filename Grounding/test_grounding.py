from datetime import datetime, timedelta
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
import os
import copernicusmarine
from pprint import pprint

with open("LENNY/Copernicus.txt") as f:
    text = f.read()
    username = text.split("\n")[0]
    password = text.split("\n")[1]

USERNAME = username
PASSWORD = password

# test 1
lon1 = -57.391907
lat = 72.082367
start_time = datetime(2021, 9, 22, 14, 32)

wind_model = "DATA/FOR_LENNY/WIND_MODELS/2021/ERA5/6h.10m_wind_2021.nc"
GLORYS = "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
# wave = "cmems_mod_glo_wav_my_0.2deg_PT3H-i"

### READER
# reader_wind = reader_netCDF_CF_generic.Reader(wind_model)
# reader_current = copernicusmarine.open_dataset(
#     dataset_id=GLORYS, username=USERNAME, password=PASSWORD
# )
# reader_current = reader_netCDF_CF_generic.Reader(reader_current)
# reader_wave = copernicusmarine.open_dataset(
#     dataset_id=wave, username=USERNAME, password=PASSWORD
# )
# reader_wave = reader_netCDF_CF_generic.Reader(reader_wave)
o = IcebergDrift(loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True)
# o.add_reader([reader_wind, reader_current])
o.set_config("environment:fallback:x_sea_water_velocity", 0)
o.set_config("environment:fallback:y_sea_water_velocity", 0.02)
o.set_config("environment:fallback:sea_floor_depth_below_sea_level", 100)


ts = 600
o.seed_elements(
    lon1,
    lat,
    number=1,
    time=start_time,
    draft=99.995024342745,
    sail=14.253629756572606,
    width=100,
)
o.seed_elements(
    lon1,
    lat,
    number=1,
    time=start_time,
)

o.run(
    duration=timedelta(days=2),
    time_step=ts,
    time_step_output=3600,
    outfile=os.path.join("test_openberg/Grounding/output", "test_grounding.nc"),
)

o.plot()
