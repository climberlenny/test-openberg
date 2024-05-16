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
reader_wind = reader_netCDF_CF_generic.Reader(wind_model)
reader_current = copernicusmarine.open_dataset(
    dataset_id=GLORYS, username=USERNAME, password=PASSWORD
)
reader_current = reader_netCDF_CF_generic.Reader(reader_current)
# reader_wave = copernicusmarine.open_dataset(
#     dataset_id=wave, username=USERNAME, password=PASSWORD
# )
# reader_wave = reader_netCDF_CF_generic.Reader(reader_wave)
o = IcebergDrift(loglevel=0, with_stokes_drift=True, wave_rad=False, water_profile=True)
o2 = IcebergDrift(
    loglevel=0, with_stokes_drift=True, wave_rad=False, water_profile=False
)
o.add_reader([reader_wind, reader_current])
o2.add_reader([reader_wind, reader_current])
# o.set_config("vertical_mixing:TSprofiles", True)

ts = 600
o.seed_elements(
    lon1,
    lat,
    number=2,
    time=start_time,
)
o.seed_elements(
    lon1,
    lat,
    number=2,
    time=start_time,
    draft=60,
    width=100,
)
o2.seed_elements(
    lon1,
    lat,
    time=start_time,
)
o.run(
    duration=timedelta(days=2),
    time_step=ts,
    time_step_output=3600,
    outfile=os.path.join("test_openberg/Profiles/output", "test_water_prof.nc"),
)
o2.run(
    duration=timedelta(days=2),
    time_step=ts,
    time_step_output=3600,
    outfile=os.path.join("test_openberg/Profiles/output", "test_water_noprof.nc"),
)
pprint(o)
pprint(o.elements)
o.plot(compare=o2)
