from opendrift.models.openberg import OpenBerg
import numpy as np
from datetime import datetime, timedelta
from pprint import pprint


o = OpenBerg(loglevel=20)


lon_seeds = np.linspace(-61, -59, 10)
lat_seeds = np.ones(shape=lon_seeds.shape) * 70

start = datetime(2021, 1, 1)
time = [start + timedelta(days=i) for i in range(len(lon_seeds))]
length = np.linspace(50, 250, 50)
width = length
height = length
coefs_sail = np.random.uniform(-0.1, 0.1, 50)
coefs_draft = np.random.uniform(-0.1, 0.1, 50)
sail = 1 / 9 * height + coefs_sail * height
draft = 8 / 9 * height + coefs_draft * height
print(lon_seeds)
print()
print(lat_seeds)
print()
print(time)
print()
print(draft / sail)


o.set_config("environment:fallback:x_wind", 5)
o.set_config("environment:fallback:y_wind", 5)
o.set_config("drift:max_age_seconds", 86401)
o.set_config("environment:fallback:x_sea_water_velocity", 0.4)
o.set_config("environment:fallback:y_sea_water_velocity", -1)

for lon, lat, t in zip(lon_seeds, lat_seeds, time):

    o.seed_elements(
        lon, lat, t, number=50, sail=sail, draft=draft, length=length, width=width
    )


o.run(time_step=3600, duration=timedelta(days=10))
pprint(o.elements)
o.plot()
