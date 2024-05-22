from datetime import datetime, timedelta
from opendrift.models.openberg_acc import IcebergDrift
from opendrift.readers import reader_netCDF_CF_generic
import os
from pprint import pprint


wind_model = "DATA/FOR_LENNY/Wind/wind_ERA5_1990_4_7.nc"
ocean = os.listdir("DATA/FOR_LENNY/IK_data")

### READER
reader_wind = reader_netCDF_CF_generic.Reader(
    wind_model, standard_name_mapping={"u10": "x_wind", "v10": "y_wind"}
)
reader_current = reader_netCDF_CF_generic.Reader(
    [os.path.join("DATA/FOR_LENNY/IK_data", o) for o in ocean]
)

o = IcebergDrift(loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True)
o.add_reader([reader_current, reader_wind])
# o.set_config("environment:fallback:x_sea_water_velocity", 0)
# o.set_config("environment:fallback:y_sea_water_velocity", 0.02)
# o.set_config("environment:fallback:sea_floor_depth_below_sea_level", 100)


ts = 900
# test 1
lon = 34.855
lat = 79.034
start_time = datetime(1990, 4, 25, 21, 0)
o.seed_elements(lon, lat, start_time, length=95, width=90, sail=2, draft=13)

o.run(
    duration=timedelta(days=3),
    time_step=ts,
    time_step_output=3600,
    outfile=os.path.join("test_openberg/test_IK/output", "7089_2.nc"),
)

o.plot()
# test2 with ensemble
o = IcebergDrift(loglevel=20, with_stokes_drift=True, wave_rad=False, grounding=True)
o.add_reader([reader_current, reader_wind])

o.seed_ensemble(
    (90, 1),
    (15, 1),
    (0.7, 0.25),
    (0.25, 0.3),
    lon=lon,
    lat=lat,
    time=start_time,
    numbers=(1, 1, 10, 10, 1),
)

o.run(
    duration=timedelta(days=3),
    time_step=ts,
    time_step_output=3600,
    outfile=os.path.join("test_openberg/test_IK/output", "7089_3_ensemble.nc"),
)

o.plot()
